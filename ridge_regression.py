from scipy.io import loadmat
import argparse
import numpy as np
import torch
import pathlib

def load_brain_y(subject, dataPath, TR=[3,4]):
    # Loading brain response for all viewed images and all ROIs for a specific subject and TR parameter.
    if isinstance(TR, list):
        return [loadmat(dataPath / "data" / "ROIS" / "CSI{}".format(subject) / "mat" / "CSI{}_ROIs_TR{}.mat".format(subject, tr)) for tr in TR]
    elif isinstance(TR, int):
        return [loadmat(dataPath / "data" / "ROIS" / "CSI{}".format(subject) / "mat" / "CSI{}_ROIs_TR{}.mat".format(subject, tr))]
    else:
        raise NotImplementedError

def extract_brain_index(stim_list, dataset="all", rep=False):
    # Code adapted from https://github.com/ariaaay/NeuralTaskonomy

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
    brain_response_idx = extract_brain_index(stimulus_list, dataset="imageNet") # this index a subset of brain response corresponding to the selected stimuli
    stimulus_to_order = {}
    img_order = np.load(dataPath / "data" / "image_order_index.npz")['arr_0']
    for i, name in enumerate(img_order):
        stimulus_to_order[name] = i
    stimulus_tensor = torch.load(dataPath / "data" / augmentType / "x{}.pt".format(layer))
    stimulus_list = np.array(stimulus_list)[brain_response_idx] # select the subset of viewed stimuli that correspond to ImageNet
    X = []
    for image_name in stimulus_list:
        X.append(stimulus_tensor[stimulus_to_order[image_name], :])
    return np.array(X), brain_response_idx


# def get_features(subj, model, layer=None, dim=None, dataset="", br_subset_idx=None, indoor_only=False):
#     print("Getting features for {}{}, for subject {}".format(model, layer, subj))

#     with open("../BOLD5000/CSI0{}_stim_lists.txt".format(subj)) as f:
#         sl = f.readlines()
#     stim_list = [item.strip("\n") for item in sl]

#     imgnet_idx, imgnet_cats = extract_dataset_index(sl, dataset="imagenet", rep=False)
#     scene_idx, scene_cats = extract_dataset_index(sl, dataset="SUN", rep=False)
#     COCO_idx, COCO_cats = extract_dataset_index(sl, dataset="COCO", rep=False)

#     # Load features list generated with the whole brain data. This dictionary includes: image names, valence responses,
#     # reaction time, session number, etc.
#     with open("{}/CSI{}_events.json".format(cortical_dir, subj)) as f:
#         events = json.load(f)

#     # events also has a stim list, it is same as the "stim_lists.txt"; but repetition is not indicated in the file name.

#     # if indoor_only:
#     #     with open("../BOLD5000_Stimuli/scene_indoor_cats.txt") as f:
#     #         lst = f.readlines()
#     #     indoor_scene_lst = [item.strip("\n") for item in lst]
#     #     indoor_scene_idx = [
#     #         i for i, s in enumerate(scene_cats) if s in indoor_scene_lst
#     #     ]
#     #     br_subset_idx = np.array(scene_idx)[indoor_scene_idx]
#     #     stim_list = np.array(stim_list)[br_subset_idx]

#     if (
#             dataset is not ""
#     ):  # only an argument for features spaces that applies to all
#         if dataset == "ImageNet":
#             br_subset_idx = imgnet_idx
#             stim_list = np.array(stim_list)[imgnet_idx]
#         elif dataset == "COCO":
#             br_subset_idx = COCO_idx
#             stim_list = np.array(stim_list)[COCO_idx]
#         elif dataset == "SUN":
#             br_subset_idx = scene_idx
#             stim_list = np.array(stim_list)[scene_idx]


#     if "convnet" in model or "scenenet" in model:

#         # Load order of image features output from pre-trai[ned convnet or scenenet
#         # (All layers has the same image order)
#         image_order = pickle.load(
#             open("../outputs/convnet_features/convnet_image_orders_fc7.p", "rb")
#         )
#         image_names = [im.split("/")[-1] for im in image_order]

#         # Load Image Features
#         if model == "convnet":
#             if "conv" in layer:
#                 feature_path = "../outputs/convnet_features/vgg19_avgpool_{}.npy".format(
#                     layer
#                 )
#             else:
#                 feature_path = "../outputs/convnet_features/vgg19__{}.npy".format(layer)
#         # elif model == 'convnet_pca':
#         #     if 'conv' in layer:
#         #         feature_path = glob("../outputs/convnet_features/*eval_pca_black_{}.npy".format(layer))[0]
#         #     else:
#         #         feature_path = glob("../outptus/convnet_features/*eval_v2_{}*.npy".format(layer))[0]
#         elif model == "scenenet":
#             feature_path = "../outputs/scenenet_features/avgpool_{}.npy".format(layer)
#         else:
#             print("model is undefined: " + model)

#         feature_mat = np.load(feature_path)
#         assert len(image_order) == feature_mat.shape[0]

#         featmat = []
#         for img_name in stim_list:
#             if "rep_" in img_name:
#                 continue  # repeated images are NOT included in the training and testing sets
#             # print(img_name)
#             feature_index = image_names.index(img_name)
#             featmat.append(feature_mat[feature_index, :])
#         featmat = np.array(featmat)

#         if br_subset_idx is None:
#             br_subset_idx = get_nonrep_index(stim_list)

#     elif "taskrepr" in model:
#         # latent space in taskonomy, model should be in the format of "taskrep_X", e.g. taskrep_curvature
#         task = "_".join(model.split("_")[1:])
#         repr_dir = "../genStimuli/{}/".format(task)
#         if indoor_only:
#             task += "_indoor"

#         featmat = []
#         for img_name in stim_list:
#             if "rep_" in img_name:
#                 # print(img_name)
#                 continue
#             npyfname = img_name.split(".")[0] + ".npy"
#             repr = np.load(repr_dir + npyfname).flatten()
#             featmat.append(repr)
#         featmat = np.array(featmat)

#         if br_subset_idx is None:
#             br_subset_idx = get_nonrep_index(stim_list)
#         print(featmat.shape[0])
#         print(len(br_subset_idx))
#         assert featmat.shape[0] == len(br_subset_idx)

#     elif model == "pic2vec":
#         # only using ImageNet
#         from gensim.models import KeyedVectors

#         wv_model = KeyedVectors.load(
#             "../outputs/models/pix2vec_{}.model".format(dim), mmap="r"
#         )
#         pix2vec = wv_model.vectors
#         wv_words = list(wv_model.vocab)
#         br_subset_idx, wv_idx = find_overlap(
#             imgnet_cats, wv_words, imgnet_idx, unique=True
#         )
#         assert len(br_subset_idx) == len(wv_idx)
#         featmat = pix2vec[wv_idx, :]

#     elif model == "fasttext":
#         # import gensim.downloader as api
#         # model = api.load('fasttext-wiki-news-subwords-300')
#         from gensim.models import KeyedVectors

#         ft_model = KeyedVectors.load("../features/fasttext.model", mmap="r")
#         if dataset == "SUN":
#             cats = scene_cats
#             idxes = scene_idx
#         elif dataset == "ImageNet":
#             cats = imgnet_cats
#             idxes = imgnet_idx
#         elif dataset == "":
#             cats = imgnet_cats + scene_cats
#             idxes = imgnet_idx + scene_idx

#         featmat, br_subset_idx = [], []
#         for i, c in enumerate(cats):
#             word = c.split(".")[0]
#             if "_" in word:
#                 word = (
#                     re.sub(r"(.)([A-Z])", r"\1-\2", word).replace("_", "-").lower()
#                 )  # convert phrases to "X-Y"
#             try:
#                 featmat.append(ft_model[word])
#                 br_subset_idx.append(idxes[i])
#             except KeyError:
#                 if "-" in word:
#                     word = word.split("-")[0]  # try to find only "X"
#                     try:
#                         featmat.append(ft_model[word])
#                         br_subset_idx.append(idxes[i])
#                     except KeyError:
#                         continue
#                 else:
#                     continue

#         featmat = np.array(featmat)
#         assert featmat.shape[0] == len(br_subset_idx)

#     elif model == "response":
#         featmat = np.array(events["valence"]).astype(np.float)
#         featmat = featmat.reshape(len(featmat), 1)  # make it 2 dimensional

#     elif model == "RT":
#         featmat = np.array(events["RT"]).astype(np.float)
#         featmat = featmat.reshape(len(featmat), 1)  # make it 2 dimensional

#     elif model == "surface_normal_latent":
#         sf_dir = "../genStimuli/rgb2sfnorm/"
#         # load the pre-trained weights
#         model_file = "../outputs/models/conv_autoencoder.pth"

#         sf_model = Autoencoder()
#         checkpoint = torch.load(model_file)
#         sf_model.load_state_dict(checkpoint)
#         sf_model.to(device)
#         for param in sf_model.parameters():
#             param.requires_grad = False
#         sf_model.eval()

#         featmat = []
#         for img_name in stim_list:
#             if "rep_" in img_name:
#                 continue
#             img = Image.open(sf_dir + img_name)
#             inputs = Variable(preprocess(img).unsqueeze_(0)).to(device)
#             feat = sf_model(inputs)[1]
#             featmat.append(feat.cpu().numpy())
#         featmat = np.squeeze(np.array(featmat))

#         if br_subset_idx is None:
#             br_subset_idx = get_nonrep_index(stim_list)

#         assert featmat.shape[0] == len(br_subset_idx)

#     elif model == "surface_normal_subsample":
#         sf_dir = "../genStimuli/rgb2sfnorm/"

#         featmat = []
#         for img_name in stim_list:
#             if "rep_" in img_name:
#                 continue
#             img = Image.open(sf_dir + img_name)
#             inputs = Variable(preprocess(img).unsqueeze_(0))
#             k = pool_size(inputs.data, 30000, adaptive=True)
#             sub_sf = (
#                 nn.functional.adaptive_avg_pool2d(inputs.data, (k, k))
#                     .cpu()
#                     .flatten()
#                     .numpy()
#             )
#             featmat.append(sub_sf)

#         featmat = np.squeeze(np.array(featmat))

#         if br_subset_idx is None:
#             br_subset_idx = get_nonrep_index(stim_list)

#         assert featmat.shape[0] == len(br_subset_idx)

#     else:
#         raise NameError("Model not found.")

#     return featmat, br_subset_idx





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="System identification with data augmentation"
    )
    # Data
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

    # load brain response
    brain_data = load_brain_y(args.subj, cur_path)
    # print(brain_data[0][])
    print(brain_data[0].keys())
    print(type(brain_data[0]["RHPPA"]))

    # load processed stimuli
    stimuli_repr, brain_response_idx = load_stimuli_x(args.layer, args.subj, cur_path, augmentType="ReducedImageNetFeaturesFlipHorizontal")

    print(stimuli_repr.shape)

