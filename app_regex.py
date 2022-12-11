import os
import gc
import io
import cv2
import base64
import pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from datasets import load_dataset
from streamlit_drawable_canvas import st_canvas
import pytesseract
from skimage.filters import threshold_local

from test_all_regex import test_regex

import time

import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

# 2022-12-09 add by @Jiaen LIU for Regex to get the date and total amount
# Regex parts are done by @Ivan STEPANIAN

@st.cache(allow_output_mutation=True)
def load_model(num_classes=2, model_name="mbv3", device=torch.device("cpu")):
    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C_aux_e3_pretrain.pth")
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")

    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    _ = model(torch.randn((1, 3, 384, 384)))

    return model


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# Add a scanner effect to the image.
def add_scanner_effect(img, parameters={
    "blur": (5,5),
    "blur_iterations": 1,
    "erode_kernel": np.ones((2,2), np.uint8),
    "erode_iterations": 1,
    "dilate_kernel": np.ones((2,2), np.uint8),
    "dilate_iterations": 1,
    "erode_kernel_2": np.ones((3,3), np.uint8),
    "erode_iterations_2": 1,
    "dilate_kernel_2": np.ones((3,3), np.uint8),
    "dilate_iterations_2": 1,
    "threshold_local_block_size": 21,
    "threshold_local_offset": 5,
    # "threshold_local_mode": 'reflect',
    "sharpen_kernel": np.array([[0,-1,0], [-1,9,-1], [0,-1,0]]),
}):
    # Convert the image to grayscale.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to the image.
    img = cv2.GaussianBlur(img, parameters["blur"], parameters["blur_iterations"])

    img = cv2.erode(img, parameters["erode_kernel"], iterations=parameters["erode_iterations"])
    img = cv2.dilate(img, parameters["dilate_kernel"], iterations=parameters["dilate_iterations"])

    # Apply a threshold to the image.
    T = threshold_local(img, parameters["threshold_local_block_size"], offset=parameters["threshold_local_offset"], method="gaussian")
    img = (img > T).astype("uint8") * 255
    
    # img = cv2.erode(img, parameters["erode_kernel_2"], iterations=parameters["erode_iterations_2"])
    # img = cv2.dilate(img, parameters["dilate_kernel_2"], iterations=parameters["dilate_iterations_2"])

    # Sharpen
    img = cv2.filter2D(src=img, ddepth=-1, kernel=parameters["sharpen_kernel"])
    return img


def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    # Preprocessing transforms. Convert to tensor and normalize.
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms


def order_points(pts):
    """Rearrange coordinates to order:
    top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)

    # Sort by Y position (to get top-down)
    pts = pts[np.argsort(pts[:, 1])]

    rect[0] = pts[0] if pts[0][0] < pts[1][0] else pts[1]
    rect[1] = pts[1] if pts[0][0] < pts[1][0] else pts[0]
    rect[2] = pts[2] if pts[2][0] > pts[3][0] else pts[3]
    rect[3] = pts[3] if pts[2][0] > pts[3][0] else pts[2]
    # s = pts.sum(axis=1)
    # # Top-left point will have the smallest sum.
    # rect[0] = pts[np.argmin(s)]
    # # Bottom-right point will have the largest sum.
    # rect[2] = pts[np.argmax(s)]

    # diff = np.diff(pts, axis=1)
    # # Top-right point will have the smallest difference.
    # rect[1] = pts[np.argmin(diff)]
    # # Bottom-left will have the largest difference.
    # rect[3] = pts[np.argmax(diff)]
    # # return the ordered coordinates
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.

    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    # if maxWidth > maxHeight:
    #     destination_corners = [[0, 0], [maxHeight, 0], [maxHeight, maxWidth], [0, maxWidth]]

    
    return order_points(destination_corners)




def scan(image_true=None, trained_model=None, image_size=384, BUFFER=10):
    """Scan the image and return the scanned image
    Args:
        image_true (np.array): Image to be scanned
        trained_model (torch.nn.Module): Trained model
        image_size (int): Size of the image to be fed to the model
        BUFFER (int): Buffer to be added to the image
    Returns:
        scanned_image (np.array): Scanned image
    """
    global preprocess_transforms

    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    # Resizing the image to the size of input to the model. (384, 384)
    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    # Converting the image to tensor and normalizing it.
    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    with torch.no_grad():
        # Out: the output of the model
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    # Garbage collection
    gc.collect()

    # Edge Detection.
    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Finding the largest contour. (Assuming that the largest contour is the document)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # ==========================================
    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    # check if corners are inside.
    # if not find smallest enclosing box, expand_image then extract document
    # else extract document

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)
        #     box_corners = minimum_bounding_rectangle(corners)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        # Find corner point which doesn't satify the image constraint
        # and record the amount of shift required to make the box
        # corner satisfy the constraint
        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER

        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER

        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER

        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        # new image with additional zeros pixels
        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)

        # adjust original image within the new 'image_extended'
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        # shifting 'box_corners' the required amount
        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)

    return final


# Generating a link to download a particular image file.
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# OCR the document by using the pytesseract library.
def ocr_document(image, lang="fra", width=600):
    imW = image.shape[1] # Get the size of the image
    if imW > width:
        # Resize the image to the width of 600 if the width is greater than 600 do that.
        image = image_resize(image, width=width)
    # Add a scanner effect to the image
    
    start_time = time.time()
    # image = add_scanner_effect(image)
    end_time = time.time()
    print("Scanner effect time: ", end_time - start_time)


    data = pytesseract.image_to_data(image, lang=lang,output_type="dict")
    words = data["text"]

    # remove empty strings
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]

    words = " ".join(words)

    print("Tesseract time: ", end_time - start_time)

    return words ,image

# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()


IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()
image = None
final = None
result = None

st.set_page_config(initial_sidebar_state="collapsed")

# processor,layoutLMv2 = get_layoutlmv2()
# id2label, label2color = get_labels()

st.title("Document Scanner: Semantic Segmentation using DeepLabV3-PyTorch, OCR using PyTesseract and LayoutLMv2 for NER")

uploaded_file = st.file_uploader("Upload Document Image :", type=["png", "jpg", "jpeg"])

language_dict = {"French": "fra", "English": "eng", "German": "deu", "Spanish": "spa", "Italian": "ita", "Dutch": "nld", "Portuguese": "por", "Russian": "rus", "Turkish": "tur", "Chinese": "chi_sim", "Japanese": "jpn", "Korean": "kor"}

method = st.radio("Select Document Segmentation Model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)

lang = st.selectbox("Select Language for OCR:", list(language_dict.keys()), index=0)



col1, col2,col3 = st.columns((6, 5,5))

if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    h, w = image.shape[:2]

    if method == "MobilenetV3-Large":
        model = load_model(model_name="mbv3")
    else:
        model = load_model(model_name="r50")

    with col1:
        st.title("Input")
        st.image(image, channels="BGR", use_column_width=True)

    with col2:
        st.title("Scanned")
        final = scan(image_true=image, trained_model=model, image_size=IMAGE_SIZE)
        st.image(final, channels="BGR", use_column_width=True)

    if final is not None:
        with col3:
            # OCR the document
            start_time = time.time()
            result,scanned_img = ocr_document(final, lang = language_dict[lang])
            # result,scanned_img = ocr_document(image, lang = language_dict[lang])
            end_time = time.time()
            print("OCR Time: ", end_time - start_time)
            st.image(scanned_img,channels="BGR", use_column_width=True)
            st.title("OCR Output")
            st.write(result)
            st.title("Regex Output")
            date, total = test_regex(result)
            st.write("Date: ", date)
            st.write("Total: ", total)
            result = Image.fromarray(final[:, :, ::-1])
            st.markdown(get_image_download_link(result, "output.png", "Download " + "Output"), unsafe_allow_html=True)
