
#  Art Generation with Neural Style Transfer

This repository is an implementation of a popular paper ([Gatys et al., 2015](https://arxiv.org/abs/1508.06576)) that demonstrates how to use neural networks to transfer artistic style from one image onto another. The application is deployed in Streamlit and do following:
- take 2 input images: content image and style image
- combine two images to generate novel artistic image

## Acknowledgements

Usefull resources
* [Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style][1]
* [Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition MatConvNet][2]

Inspired by following projects
* [TensorFlow Implementation of "A Neural Algorithm of Artistic Style"][3]
* [Harish Narayanan, Convolutional neural networks for artistic style transfer.][4]

[1]: https://arxiv.org/abs/1508.06576
[2]: https://arxiv.org/pdf/1409.1556.pdf
[3]: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
[4]: https://harishnarayanan.org/writing/artistic-style-transfer/

## Demo (click to view video)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/iJvPCGDfUqs/0.jpg)](https://www.youtube.com/watch?v=iJvPCGDfUqs)


## Run Locally

Clone the project

```bash
  git clone https://github.com/NurNazaR/Art-generation-with-neural-style-transfer.git
```

Go to the project directory

```bash
  cd "Art generation with neural style transfer"
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Upload your content and style images to the 'images' folder 

Run the art style transfer application

```bash
  cd app
```

```bash
  streamlit run style_app.py 
```

```


## License

[MIT](https://choosealicense.com/licenses/mit/)

