# GroundingDINO & SAM Project

This project integrates GroundingDINO with Segment Anything Model (SAM) to detect and annotate objects such as stoplights and signs in videos. The results are saved as annotated frames and JSON files containing detection details.
You can find an introduction to the concepts in the <a href="https://arxiv.org/abs/2303.05499">research paper</a>.
<img src="https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GD_GLIGEN.png" alt="gd_gligen" width="100%">

- Grounding DINO accepts an `(image, text)` pair as inputs.
- It outputs `900` (by default) object boxes. Each box has similarity scores across all input words.
- Boxes with the highest similarities above a `box_threshold` are selected.
- Words with similarities above the `text_threshold` are extracted as predicted labels.
- To obtain objects of specific phrases, such as `dogs` in the sentence `two dogs with a stick.`, select the boxes with the highest text similarities to `dogs`.
- Each word can be split into multiple tokens, so the number of words may not equal the number of text tokens.
- Separate different category names with `.` for Grounding DINO.
![model_explain1](.asset/model_explain1.PNG)
![model_explain2](.asset/model_explain2.PNG)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
    pip install -e .
    ```

2. Install additional dependencies:

    ```bash
    pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    pip install supervision==0.12.0
    ```

3. Download the weights:

    ```bash
    mkdir -p weights
    cd weights
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

## Usage

### Running the Script

1. Place your video file in the project directory.
2. Update the `video_path` variable in `test.py` with the name of your video file.

    ```python
    video_path = "your_video_file.mp4"
    ```

3. Run the `test.py` script to process the video:

    ```bash
    python test.py
    ```

The script will:
- Read the video frames.
- Detect objects (stoplights and signs) using GroundingDINO.
- Annotate the frames with bounding boxes and labels.
- Save the annotated frames and detection results in a JSON file in the `res_<video_name>` directory.

Here are some sample outputs from the project:

![Sample Image 1](./img/frame_110.png)
![Sample Image 2](./img/frame_1870.png)