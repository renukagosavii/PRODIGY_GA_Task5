Neural Style Transfer – Task 05

Project: Apply the artistic style of one image to the content of another using Neural Style Transfer.
Author: Comillas Negras / ProDigy Infotech

content of another using Neural Style Transfer.
Author: Comillas Negras / ProDigy Infotech

Overview
Neural Style Transfer (NST) is a technique that blends the content of one image with the style of another using convolutional neural networks.
Content Image: Preserves the original structure (e.g., a circle).
Style Image: Provides artistic features (e.g., painting style).
Output Image: Combination of content and style.
This project uses PyTorch and a pre-trained VGG19 network to extract features and apply style transfer.


Project Structure :-
Neural-Style-Transfer-Task05/
│
├── images/
│   ├── circle.jpg          # Content image
│   ├── starry_night.jpg    # Style image
│   └── output.jpg          # Generated stylized output
│
├── scripts/
│   └── neural_style_transfer_task5_fixed.py  # Main script
│
├── README.md
├── requirements.txt        # Dependencies
└── .gitignore

Dependencies :-
Python 3.10+
PyTorch
torchvision
Pillow
Install dependencies:

pip install -r requirements.txt
Setup Instructions
Place your content and style images in the images/ folder.
Ensure the images are valid and non-empty.
Update paths in the script if needed:

content_path = r"images/circle.jpg"
style_path   = r"images/starry_night.jpg"

Running the Script
Open a terminal in the project folder.
Run the style transfer script:

The script prints progress every 50 iterations.
The output is saved as images/output.jpg.
python scripts/neural_style_transfer_task5_fixed.py
Algorithm Documentation
1. Feature Extraction
Uses a pre-trained VGG19 network.
Extracts content features from conv4_2.
Extracts style features from conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.

2. Gram Matrix for Style
Computes the Gram matrix of style features.
Captures texture and style patterns independent of spatial arrangement.

3. Optimization
Initializes the target image as a copy of the content image.
Uses Adam optimizer to minimize combined loss:

Loss
=
𝛼
⋅
Content Loss
+
𝛽
⋅
Style Loss
Loss=α⋅Content Loss+β⋅Style Loss
Content Loss: Mean squared error between target and content features.
Style Loss: Weighted sum of MSE between target and style Gram matrices.

4. Output
The target image is updated iteratively.
Final result is saved as output.jpg.

Example Output

Content: circle.jpg
Style: starry_night.jpg
Output: output.jpg

Notes
Script automatically uses GPU if available, otherwise CPU.
Recommended image size: <500px for faster processing.
Empty or corrupted images will cause runtime errors.
You can adjust weights and iterations in the script for different effects.

References

Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks.
PyTorch Documentation: https://pytorch.org/
This README is professional, detailed, and ready for GitHub, including project overview, setup, algorithm explanation, and usage instructions.
If you want, I can also create a GitHub-ready repository folder structure with this README, scripts, and placeholder images, so you can push it directly without any setup issues.
