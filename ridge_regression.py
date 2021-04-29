import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pathlib
import pickle
import math
from scipy.io import loadmat
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    PredefinedSplit,
    train_test_split,
    GroupShuffleSplit,
    ShuffleSplit,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from ridge import RidgeCVEstimator

"""
Code adapted from https://github.com/ariaaay/NeuralTaskonomy Author: Aria Wang et.al.
"""

ROIS = ["OPA", "PPA", "LOC", "EarlyVis", "RSC"]
SIDE = ["LH", "RH"]

def load_brain_y(subject, dataPath, TR=[3,4]):
    # Loading brain response for all viewed images and all ROIs for a specific subject and TR parameter.
    if isinstance(TR, list):
        return [loadmat(dataPath / "data" / "ROIS" / "CSI{}".format(subject) / "mat" / "CSI{}_ROIs_TR{}.mat".format(subject, tr)) for tr in TR]
    elif isinstance(TR, int):
        return [loadmat(dataPath / "data" / "ROIS" / "CSI{}".format(subject) / "mat" / "CSI{}_ROIs_TR{}.mat".format(subject, tr))]
    else:
        raise NotImplementedError

def extract_brain_index(stim_list, dataset="all", rep=False):
    dataset_labels = []
    COCO_idx, imagenet_idx, SUN_idx = list(), list(), list()
    for i, n in enumerate(stim_list):
        if "COCO_" in n:
            dataset_labels.append("COCO")
            if "rep_" in n and rep is False:
                continue
            COCO_idx.append(i)
        elif "JPEG" in n:  # imagenet
            dataset_labels.append("imagenet")
            if "rep_" in n:
                if not rep:
                    continue
            imagenet_idx.append(i)
        else:
            dataset_labels.append("SUN")
            if "rep_" in n:
                if not rep:
                    continue
            SUN_idx.append(i)
    if rep:
        assert len(stim_list) == len(COCO_idx) + len(imagenet_idx) + len(SUN_idx)

    if dataset == "COCO":
        return COCO_idx
    elif dataset == "imagenet":
        return imagenet_idx
    elif dataset == "SUN":
        return SUN_idx
    else:
        return dataset_labels

def load_stimuli_x(layer, subject, dataPath, augmentType):
    print("loading stimuli from layer {} of subject{}".format(layer, subject))

    with open(dataPath / "data" / "ROIS" / "stim_lists" / "CSI0{}_stim_lists.txt".format(subject)) as f:
        stimulus_list = [item.strip("\n") for item in f.readlines()]
    brain_response_idx = np.array(extract_brain_index(stimulus_list, dataset="imagenet")) # this index a subset of brain response corresponding to the selected stimuli
    stimulus_to_order = {}
    img_order = np.load(dataPath / "data" / "image_order_index.npz")['arr_0']
    for i, name in enumerate(img_order):
        stimulus_to_order[name] = i
    stimulus_tensor = np.load(dataPath / "data" / augmentType / "x{}.npy".format(layer))
    stimulus_list = np.array(stimulus_list)[brain_response_idx] # select the subset of viewed stimuli that correspond to ImageNet
    X = []
    for image_name in stimulus_list:
        X.append(stimulus_tensor[stimulus_to_order[image_name], :])
    return np.array(X), brain_response_idx

def parse_br(brain_response, br_index, side, roi):
    #print(type(brain_response[0][side + roi]))
    #print(brain_response[0][side + roi].shape)
    #print(br_index.shape)
    if len(brain_response) == 1:
        return brain_response[0][side + roi][br_index, :]
    else:
        return (brain_response[0][side + roi][br_index, :] + brain_response[1][side + roi][br_index, :]) / 2

def ridge_regression_with_cv(stimuli ,brain_response, cv=True, seed=42, ridge_lambdas = None, test_size=0.15, n_fold=7):
    
    performance_measure = lambda y, y_pred: -F.mse_loss(y_pred, y)
    ridge_lambdas = torch.from_numpy(np.logspace(-8, 1 / 2 * np.log10(stimuli.shape[1]) + 8, 100)) if not ridge_lambdas else torch.from_numpy(ridge_lambdas)
    stimuli_train, stimuli_test, br_train, br_test = train_test_split(stimuli, brain_response, test_size=test_size, random_state=seed)

    to_torch = lambda x: torch.from_numpy(x).to(dtype=torch.float64)
    stimuli_train, stimuli_test, br_train = to_torch(stimuli_train), to_torch(stimuli_test), to_torch(br_train)
    if cv:
        kfold = KFold(n_splits=n_fold)
    else:
        train_index, _ = next(ShuffleSplit(test_size=test_size).split(stimuli_train, br_train))
        # manually construct a train/val split
        split_index = np.zeros(stimuli_train.shape[0])
        split_index[train_index] = -1
        kfold = PredefinedSplit(split_index)
        assert kfold.get_n_splits() == 1
    
    ridgeModel = RidgeCVEstimator(ridge_lambdas, kfold, performance_measure, scale_X=False)
    print("ridge regression model instance created")
    print("stimuli_train shape: ", stimuli_train.shape)
    print("brain response train shape: ", br_train.shape)
    ridgeModel.fit(stimuli_train, br_train)
    br_pred = ridgeModel.predict(stimuli_test).cpu().numpy()
    r_squares = [r2_score(br_test[:,i], br_pred[:,i]) for i in range(br_test.shape[1])]
    print("r_square stat: min {}, max {}, mean {}".format(min(r_squares), max(r_squares), sum(r_squares) / len(r_squares)))
    correlations = [pearsonr(br_test[:,i], br_pred[:,i]) for i in range(br_test.shape[1])]
    correlations_val = [item[0] for item in correlations]
    print("pearson correlation stat: min {}, max {}, mean {}".format(min(correlations_val), max(correlations_val), sum(correlations_val) / len(correlations_val)))

    return correlations, r_squares, ridgeModel.mean_cv_scores.cpu().numpy(), ridgeModel.best_l_scores.cpu().numpy(), ridgeModel.best_l_idxs.cpu().numpy(), [br_pred, br_test]

def experiment(stimuli, brain_response, brain_response_index, outputPath, cv=True, export=True, ROI="all", permute_y=False):
    # ROI: list of all interested ROI or defaulted to All
    selected_ROIS = ROIS if ROI == "all" else ROI
    d_correlations, d_r_squares, d_mean_cv_scores, d_best_l_scores, d_best_l_idxs, d_predictions = {}, {}, {}, {}, {}, {}
    for side in SIDE:
        for roi in selected_ROIS:
            print("doing experiment for {} and {}".format(side, roi))
            brain_response_at_roi = parse_br(brain_response, brain_response_index, side, roi)
            if permute_y:
                print("\TODO: add permutation test support")
                break
            correlations, r_squares, mean_cv_scores, best_l_scores, best_l_idxs, predictions = ridge_regression_with_cv(stimuli, brain_response_at_roi, cv)
            d_correlations[side+roi] = correlations
            d_r_squares[side+roi] = r_squares
            d_mean_cv_scores[side+roi] = mean_cv_scores
            d_best_l_scores[side+roi] = best_l_scores
            d_best_l_idxs[side+roi] = best_l_idxs
            d_predictions[side+roi] = predictions # predictions => [prediction, ground truth]

    if export:
        pickle.dump(d_correlations, open(outputPath / "correlations.pk", "wb"))
        pickle.dump(d_r_squares, open(outputPath / "r_squares.pk", "wb"))
        pickle.dump(d_mean_cv_scores, open(outputPath / "mean_cv_scores.pk", "wb"))
        pickle.dump(d_best_l_scores, open(outputPath / "best_l_scores.pk", "wb"))
        pickle.dump(d_best_l_idxs, open(outputPath / "best_l_idxs.pk", "wb"))
        pickle.dump(d_predictions, open(outputPath / "predictions.pk", "wb"))

    print("experiment done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="System identification with data augmentation"
    )
    # Data
    parser.add_argument(
        "--exp-name",
        type=str,
        default="default",
        help="experiment name use to initialize log folder."
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=3,
        help="layer number of Alexnet for producing feature space. Choose from [2,5,7,9,12]"
    )
    parser.add_argument(
        "--subj",
        type=int,
        default=1, 
        help="subject number to build encoding model"
    )
    parser.add_argument(
        "--TR", type=str, default="NOTSUPPORTED", help="specify which TR to build model on" #\TODO figure out how to use TR parameter
    )
    # Encoding model
    parser.add_argument(
        "--noPermutation",
        action="store_true",
        default=True,
        help="run ridge regression without permutation test"
    )
    parser.add_argument(
        "--PermutationTest",
        action="store_true",
        default=False,
        help="running permutation testing only"
    )
    parser.add_argument(
        "--cv", action="store_true", default=False, help="run cross-validation"
    )
    parser.add_argument(
        "--permute_y",
        action="store_true",
        default=False,
        help="permute test label but not training label",
    )
    parser.add_argument(
        "--dataPath",
        type=str,
        help="path to stimuli readout and brain response",
    )
    # parser.add_argument(
    #     "--dim", type=str, default="", help="specify the dimension of the pic2vec model"
    # )
    # parser.add_argument(
    #     "--stacking",
    #     type=str,
    #     default=None,
    #     help="run stacking net to select features in the joint model.",
    # )
    # parser.add_argument(
    #     "--split_by_runs",
    #     action="store_true",
    #     help="split the training and testing samples by runs.",
    # )
    # parser.add_argument(
    #     "--pca",
    #     action="store_true",
    #     help="run pca on features; has to applied on per datasets training.",
    # )
    # parser.add_argument(
    #     "--indoor_only", action="store_true", help="run models only on indoor images."
    # )
    # parser.add_argument(
    #     "--fix_testing",
    #     action="store_true",
    #     help="used fixed sampling for training and testing",
    # )
    args = parser.parse_args()

    cur_path = pathlib.Path.cwd()

    augmentType = "ReducedImageNetFeaturesFlipHorizontal"

    # load brain response
    brain_data = load_brain_y(args.subj, cur_path)

    # load processed stimuli
    stimuli_repr, brain_response_idx = load_stimuli_x(args.layer, args.subj, cur_path, augmentType=augmentType)

    outputPath = cur_path / "logs" / args.exp_name / "sub{}_layer{}_augment_{}".format(args.subj, args.layer, augmentType)

    try:
        outputPath.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("folder exist -- skip")
    else:
        print("New output path folder created")

    experiment(stimuli_repr, brain_data, brain_response_idx, outputPath, ROI=["EarlyVis"]) # delete the ROI argument to get all ROIs' prediction
        
