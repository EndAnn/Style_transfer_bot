# Style-Transfer-Telegram-Bot

link https://t.me/train_anna_bot
The bot runs from a remote server

## About bot
This Telegram bot transfers the style of one picture to another, using the Neural Style Transfer algorithm. This method allows to combine the content of one image with the aesthetics and style of another, creating unique artificial images.

The Мodel is based on
 https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

The example of it's work:

![](https://github.com/EndAnn/Style_transfer_bot/blob/main/images/gith1.jpg)


## Work of bot
The bot perceives two commands:

- **/start**- Start working with the bot.
Send two images with the captions "content" and "style".
The bot also waits for pictures with style and content tags.

- **/transfer_style** - Start the style transfer process.
Wait for the process to complete, the bot will send the result.

If the user sends unfamiliar commands, or pictures with wrong labels, the bot informs the user about it and asks to do everything as it should.

## How to start a project

### Run on your local machine
- Copy the files to any folder.
- Run the script main_code_style_transfering_bot.py
The TG_BOT_TOKEN.txt file should contain your token for the bot.

You can choose
- the size of the smaller side of the output image --size (default=256)
- in the file style_trasfering.py , in the class Style_Transfer, in the function transferring there is a restriction on the size of the output image (128) in case of absence of cuda.
- number of optimizer steps --num_steps (default=25)
- weight of the style loss function --style_weight (default=70000)
(when the weight is reduced, the influence of style is also reduced)
 weight of the content loss function --content_weight (default=1)

values of loss functions should decrease during the learning process

### About code
Written using the libraries 'python-telegram-bot', 'torch'.
The main code is in the file app.py.
This is a generic file combining style transfer(def draw) and telegram bot.

File style_trasfering.py contains contains all the processes involved in image transfering - reading and converting images, the style transfer itself, the image converting, logs saving

File model_support.py contains elements of Neural Style Transfer algorithm: use of weights of pre-trained neural network, calculation of loss function, optimization of loss function and parameters of source image

During its work the bot creates a folder badger_style_transferring_bot, as well as separate folders for each chat where it stores temporary image files and losses. After receiving the output, the image files are deleted. The code with cell outputs is saved to the file style_transfer.log.

Layer weights are initialized by the weights of the VGG model in the vgg_weights.pth file.

## References
 <a id="1">[1]</a>
 https://stepik.org/course/122947/syllabus

 <a id="2">[2]</a>
 https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

 <a id="3">[3]</a>
  "A Neural Algorithm of Artistic Style",
  Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, 2015

 <a id="4">[4]</a>
 https://www.kaggle.com/code/shashank069/neuralstyletransfer-experiments/notebook
