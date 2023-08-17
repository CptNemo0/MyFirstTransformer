<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/CptNemo0/MyFirstTransformer">
    <img src="images/logo.jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Transformer Decoder Implementation</h3>

  <p align="center">
    My 'tiny-scale' implementation of decoder (ChatGPT)
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Description/Docs</strong></a>
    <br />
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
<!--
[![Product Name Screen Shot][product-screenshot]](https://example.com)-->

I have spent most of last 2 summers (2022, 2023) studying artificial intelligence. For the past 2 months ([6/7]/2023) I was studying deep learning, especially very popular transformer architecture. This project is inauguration of my studies. It's simple, small scale implementation of the decoder part of the transformer. 

It can be trained on .txt files. This is very primitive approach, nontheless it results in visible results. I have trained a small 100M model on old polish texts, and the output was striking. It wasn't perfectly logical, or beautiful, but:
* it was polish
* it sounded like archaic polish (which was the goal)
* proof that my code works
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

I used Python, and it's machine learning library PyTorch, although it's important to notice, that i didn't use stock implmentation for everything, mainly so I could get more expirience and a better understanding.

* [![Python][Python]][Python-url]
* [![PyTorch][Pytorch]][Pytorch-url]
* [⏳]([TikToken-url]) tiktoken
<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Installation

To tinker with my code you will first need to clone to repository
   ```sh
   git clone https://github.com/CptNemo0/MyFirstTransformer
   ```
Than install requirements from the txt file
   ```sh
   pip install -r requirements.txt
   ```
You're good to go!!
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Work in progress. TBA

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] Barebones logic
- [x] Working small scale models
- [x] Designed gui
- [ ] Large Language Model 
- [ ] Research more pretrainig techniques
- [x] Research finetuning (chain-of-thought, intruction finetuning, meta-learning (few shot))
- [ ] Working gui
    - [ ] Training gui
    - [ ] Inferece gui
- [ ] Documentation
- [ ] Usage section of README

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Paweł Stus - pawel.j.stus@gmail.com 
Project Link: https://github.com/CptNemo0/MyFirstTransformer
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Resources
#### Scientific
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
* [MIT Transformer Course](https://web.stanford.edu/class/cs25/) - especially scientific papers listed in the "Recommended Reading" sections
* [MIT Intro to deep learning course](http://introtodeeplearning.com/)
* [Andrej Karpathy](https://karpathy.ai/) aka the "Big Boss" 
* [OpenAI](https://openai.com/)

#### Others
* [README](https://github.com/othneildrew/Best-README-Template)
* [Img Shields](https://shields.io)
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/downloads/release/python-3114/

[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=for-the-badge
[Pytorch-url]: https://pytorch.org/

[TikToken]: ⏳
[TikToken-url]: https://github.com/openai/tiktoken
