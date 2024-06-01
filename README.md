# Automation

# MCQ Bubble Sheet Grader

This project is a Python script for grading multiple-choice question (MCQ) bubble sheets using OpenCV. The script processes an input image of a bubble sheet, detects the filled-in answers, compares them to an answer key, and calculates the score.

## Requirements

- Python 3.6+
- OpenCV
- imutils
- numpy

You can install the required packages using pip:

```sh
pip install opencv-python imutils numpy
```

## Usage

1. Place the image of the bubble sheet you want to grade in the same directory as the script or provide the path to the image.

2. Run the script with the `--image` argument:

```sh
python script.py --image path/to/your/image.jpg
```

3. The script will display intermediate steps including the grayscale, blurred, and edge-detected images, as well as the detected contours. It will then process the bubbles, compare them with the answer key, and finally display the graded sheet with the score.

## Answer Key

The answer key is defined in the script as `ANSWER_KEY1` and `ANSWER_KEY`. Adjust the values in `ANSWER_KEY1` if your bubble sheet has a different correct answer set.

## Example

```sh
python script.py --image bubble_sheet.jpg
```

## Notes

- Ensure the image is clear and well-aligned for accurate detection.
- The script uses Otsu's thresholding to binarize the image and detects contours to identify bubbles.

## License

This project is licensed under the MIT License.
```

Make sure to replace `script.py` with the actual name of your Python script if it's different. Save this content to a file named `README.md` in your project directory.
