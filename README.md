# Transfer Learning with EfficientNet For The Classification of Brain Tumor MR Images

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/brain-logo.png" alt="Logo" width="200" height="200">
  </a>

<h3 align="center">Transfer Learning with EfficientNet For The Classification of Brain Tumor MR Images
 <br>An End-to-End Deep Learning Project</h3>

  <p style="text-align: center;"><span style="font-weight: 400;">In this project, we developed and deployed a <strong>classifier for four different types of brain tumor MR images: Glioma, Meningioma, Pituitary and No Tumor</strong>. <br>
Utilizing the method of <strong>transfer learning</strong> and<strong> EfficientNet B3</strong>, the final model achieved an <strong>accuracy rate of 99.23%</strong></span></p>

    
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<details>
<summary> Table of Contents</summary>
<p>🧠 Table of content:</p>
<ol>
    <li>
        <p>About the project 📃</p>
    </li>
    <li>
        <p>Data Pre-Processing ✂️✂️✂️✂️✂️✂️</p>
        <ol>
            <li>
                <p>Cropping the original images</p>
            </li>
        </ol>
    </li>
    <li>
        <p>Model Developing &amp; Deploying 🛠️ 🚀</p>
        <ol>
            <li>
                <p>Make the development of applications faster by creating a series of useful functions&nbsp;</p>
            </li>
            <li>
                <p>Generate training, testing, and validation dataframes from testing and training directories (image paths and labels).</p>
            </li>
            <li>
                <p>Manage the balance of the dataset by adding augmented images to minority classes</p>
            </li>
            <li>
                <p>Generates batches of augmented/normalized data for training, testing and validation dataframes</p>
            </li>
            <li>
                <p>Display examples of training images</p>
            </li>
            <li>
                <p>Create models using transfer learning with EfficientNet (B0,B3,B5,B7)</p>
            </li>
            <li>
                <p>Custom model callback</p>
            </li>
            <li>
                <p>Train model</p>
            </li>
            <li>
                <p>Display metrics (loss and accuracy) for training and validation</p>
            </li>
            <li>
                <p>Making prediction on the test set</p>
                <ol>
                    <li>
                        <p>Confusion matrix&nbsp;</p>
                    </li>
                    <li>
                        <p>ROC &amp; AUC curves&nbsp;</p>
                    </li>
                </ol>
            </li>
            <li>
                <p>Export and deploy trained model</p>
            </li>
        </ol>
    </li>
    <li>
        <p>Conclusions!</p>
    </li>
</ol>
<p><br></p>
</details>