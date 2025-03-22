# --------------------------------------------------------
# Visual Instruction Tuning
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# LISA Questions and GSVA questions

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

SHORT_QUESTION_LIST_MODE4 = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please respond with segmentation masks.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please output segmentation masks."
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explaination.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

ANSWER_LIST_MODE1 = [
    "Here it is.",
    "Sure.",
    "Sure, this is the target.",
    "Sure, here is the segmentation result.",
    "Here you are."
]

ANSWER_LIST_MODE4_START = [
    "The segmentation results are",
    "Sure, they are",
    "Sure,",
    "Sure,",
    "Sure,"
]

ANSWER_LIST_MODE4_TEMPLATE = [
    "{class_name} [SEG]",
    "{class_name}:[SEG]",
    "the mask of {class_name} is [SEG]",
    "the segmentation of {class_name} is [SEG]",
    "the referred {class_name} is [SEG]"
]

ANSWER_LIST_MODE4_END = [
    ".", ".", ".", ".", "."
]

TAMPER_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Has the image been altered? if yes, please locate the location of the tampering.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Has this image been modified? if yes, please locate the location of the tampering.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Has this image been tampered with? if yes, please locate the location of the tampering.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Has this photo been tampered with? if yes, please locate the location of the tampering.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Is there any sign that this image has been changed? if yes, please locate the location of the tampering.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Has this image been erased or replaced? if yes, please locate the location of the tampering.",
]

EXPLANATORY_TAMPER_QUESTION_LIST = [
	"If yes, Please output tampered area mask and explain why. ",
    "Please output tampered area mask and explain the reason.",
    "Please output tampered area mask and give some explaination.",
    "If yes, please output the tampered regions and provide their explanations.",
    "If yes, please provide the tampered areas and an explanation for each identified tampering type.",
    "Please provide the tampered areas and an explanation for each identified tampering type."
]


NOTAMPER_ANSWER_LIST = [
    "No, the picture has not been altered.",
    "No, there are no signs of tampering in this picture.",
    "No, there is no indication of tampering in the image.",
    "No, this image remains unchanged.",
    "No, this image has not been forged.",
    "No, this image has not been manipulated."
]

ANSWER_START = [ 
    "Yes, this image has been tampered with,",
    "Yes, the picture has undergone manipulation,",
    "Yes, there are traces of tampering in this image,",
    "Yes, evidence of forgery can be seen in the picture,",
    "Yes, this picture has undergone editing,",
    "Yes, the image has been manipulated,",
]

ONLY_ANSWER_LIST = [
    "the tampered location is at ({center})",
    "tamper position: ({center})",
    "the position of the tampering: ({center})",
    "the tamper location lies at ({center})",
    "the location of the tampering is ({center})",
]
ONLY_ANSWER_LIST_TASK2 = [
    "the tampered location is at [NSTART] {center} [NEND]",
    "tamper position: [NSTART] {center} [NEND]",
    "the position of the tampering: [NSTART] {center} [NEND]",
    "the tamper location lies at [NSTART] {center} [NEND]",
    "the location of the tampering is [NSTART] {center} [NEND]",
]

MULTI_ANSWER_LIST = [
    "the {order} location of the tampering is ({center})",
    "the {order} tampering is located at the ({center})",
    "the {order} position of the alteration is ({center})",
    "the {order} tampered position is ({center})",
    "the {order} tampered position: ({center})",
    "{order} tampered location: ({center})"
]
MULTI_ANSWER_LIST_TASK2 = [
    "the {order} location of the tampering is [NSTART] {center} [NEND]",
    "the {order} tampering is located at the [NSTART] {center} [NEND]",
    "the {order} position of the alteration is [NSTART] {center} [NEND]",
    "the {order} tampered position is [NSTART] {center} [NEND]",
    "the {order} tampered position: [NSTART] {center} [NEND]",
    "{order} tampered location: [NSTART] {center} [NEND]"
]

ANSWER_CENTER = [
    "its location is at: {center}",
    "the position is: {center}",
    "it is situated at {center}",
    "it lies at {center}",
    "position:{center}",
    "location:{center}"
]

ANSWER_TAMPERTYPE = [
    "So, the tamper type is: {tampertype}.",
    "Its tampering type is {tampertype}.",
    "Tamper type: {tampertype}.",
    "The type of tampering for this image is {tampertype}.",
    "This image is likely tampered using the {tampertype} method.",
]

ANSWER_LIST_END = [
    ".", ".", ".", ".", ".", "."
]
ANSWER_LIST_DELAY = [
    ",", ",", ",", ",", ",", ","
]

ONLY_ANSWER_SEG_LIST = [
    "the tampered location is at [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
    "tamper position: [POT{i}] , it is {center},and its bounding box is [BOX{i}]. ",
    "the position of the tampering: [POT{i}] , it is {center},and its bounding box is [BOX{i}]. ",
    "the tamper location lies at [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
    "the location of the tampering is [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
]
MULTI_ANSWER_SEG_LIST = [
    "the {order} location of the tampering is [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
    "the {order} tampering is located at the [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
    "the {order} position of the alteration is [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
    "the {order} tampered position is [POT{i}] : {center},and its bounding box is [BOX{i}]. ",
    "the {order} tampered position: [POT{i}] , it is {center},and its bounding box is [BOX{i}]. ",
    "{order} tampered location: [POT{i}] , it is {center},and its bounding box is [BOX{i}]. "
]

ONLY_ANSWER_SEG_LIST_ABU = [
    "the tampered location is at [POT] : {center},and its bounding box is [BOX]. ",
    "tamper position: [POT] , it is {center},and its bounding box is [BOX]. ",
    "the position of the tampering: [POT] , it is {center},and its bounding box is [BOX]. ",
    "the tamper location lies at [POT] : {center},and its bounding box is [BOX]. ",
    "the location of the tampering is [POT] : {center},and its bounding box is [BOX]. ",
]
MULTI_ANSWER_SEG_LIST_ABU = [
    "the {order} location of the tampering is [POT] : {center},and its bounding box is [BOX]. ",
    "the {order} tampering is located at the [POT] : {center},and its bounding box is [BOX]. ",
    "the {order} position of the alteration is [POT] : {center},and its bounding box is [BOX]. ",
    "the {order} tampered position is [POT] : {center},and its bounding box is [BOX]. ",
    "the {order} tampered position: [POT] , it is {center},and its bounding box is [BOX]. ",
    "{order} tampered location: [POT] , it is {center},and its bounding box is [BOX]. "
]