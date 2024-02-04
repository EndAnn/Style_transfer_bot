from telegram.ext import CommandHandler, MessageHandler,Updater, Filters

import os
import logging
import argparse
import requests
from pathlib import Path
import shutil
import json
from PIL import Image
from io import BytesIO

# мои модули
import style_trasfering

########## Две вспомогательные функции ##############
# Загружает изображение по file_path
# Изменяет размер изображения до заданного size x size
# Сохраняет изображение и пропорцию в каталоге для конкретного чата под именем category.jpg и category_proportion.txt
def load_save(file_path, size , chat_id , category, work_dir):
    try :
        response  = requests.get(file_path)
        img = Image.open(BytesIO(response.content))
        proportion = img.size[0]/img.size[1]
        print ('proportion from load_save', proportion)
        img=img.resize((size, size))
        save_images = work_dir/str(chat_id)
        if not os.path.exists(save_images ):
                os.makedirs(save_images)

        img.save(save_images/(category +'.jpg'))
        with open (save_images/(category + '_proportion.txt'), 'w') as file:
            file.write(str(proportion))
        return 0
    # текст ошибки в случае неудачи
    except Exception as e :
        return str(e)

# Открывает ранее сохраненное изображение из каталога чата.
# Восстанавливает изображение с учетом сохраненной пропорции.
# Преобразует изображение в формат, который бот может отправить в Telegram.
def open_rebuilt( size, chat_id , category, work_dir ):
    save_images = work_dir /str(chat_id)
    img = Image.open(save_images /( category + '.jpg'))
    with open(save_images/(category + '_proportion.txt'), 'r') as file:
        proportion = float(file.readlines()[0])
    img = img.resize( (int(size*proportion),size))
    # Преобразует изображение в формат, который бот может отправить в Telegram.
    bio = BytesIO()
    bio.name = 'image.jpeg'
    img.save(bio, 'JPEG')
    bio.seek(0)
    return bio

parser = argparse.ArgumentParser(description='app')
parser.add_argument("--size", type=int, default=256,
                    help="размер меньшей стороны обработанного изображения")
parser.add_argument("--num_steps", type=int, default=25,
                    help="число шагов оптимайзера")
# меняя параметр --style_weight можно получить разные выходные картинки
parser.add_argument("--style_weight", type=int, default=70000,
                    help="вес функции потерь стиля")
parser.add_argument("--content_weight", type=int, default=1,
                    help="вес функции потерь контента")
opt = parser.parse_args()
print(opt)

# текущая директория
CURRENT_DIR = Path(__file__).parent.resolve()

#  TOKEN
with open(CURRENT_DIR/'TG_BOT_TOKEN.txt') as file:
    TOKEN = file.readline()

# временная рабочая  папка (фотки стиля и контента, loss и готовое фото)
work_dir_name = 'style_transfering_bot'
WORK_DIR = CURRENT_DIR / work_dir_name
Path.mkdir ( WORK_DIR, parents=True, exist_ok=True)

# трансформация
ST = style_trasfering.Style_Transfer()

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Я бот, который переносит стиль с одного изображения на другой.\n \
                              Пришли два изображения с пометками content и style, \n \
                              После чего запусти команду '/transfer_style'")

def image(update, context):
    chat_id = update.effective_chat.id
    category = update.message.caption
    if category in ['content', 'style']:
        # id и url картинки
        file_id = update.message.document.file_id
        file_path = context.bot.get_file(file_id).file_path

        result = load_save(
            file_path=file_path, size = opt.size+10,
            chat_id = chat_id, category = category,
            work_dir = WORK_DIR)

        if result:
            text = f'Что-то не то с файлом...'
            print( f'image exception  {result}' )
        else:
            text =f'{category}'
    else:
        text = "Должна быть подпись к изображению 'content'/'style' "

def photo(update, context):
    chat_id = update.effective_chat.id
    category = update.message.caption

    if category in ['content', 'style']:
        file_id = update.message.photo[-1].file_id
        file_path = context.bot.get_file(file_id).file_path
        result = load_save(
            file_path=file_path, size = opt.size+10,
            chat_id = chat_id, category = category,
            work_dir = WORK_DIR
        )
        if result:
            text = f'Что-то не то с файлом...'
            print(print( f'photo exception {result}' ))
        else:
            text =f'{category}'
    else:
        text = "Должна быть подпись к изображению 'content'/'style' "

    context.bot.send_message(chat_id = chat_id, text=text )

def transfer_style(update, context):
    chat_id = update.effective_chat.id
    path_for_save_images = WORK_DIR /str(chat_id)
    if not Path(  path_for_save_images /'content.jpg').exists():
        context.bot.send_message (chat_id=update.effective_chat.id, text= 'Пришли изображение с пометкой "content", стиль которого будет использоваться' )
    elif not Path(path_for_save_images /'style.jpg').exists():
        context.bot.send_message(chat_id=update.effective_chat.id, text= 'Пришли изображение с пометкой "style", к которому будет применен новый стиль')
    else :
        for category in ['style', 'content']:
            img = open_rebuilt(
                size = 150,
                chat_id = chat_id, category = category,
                work_dir = WORK_DIR
            )
            context.bot.send_photo(chat_id=chat_id, photo=img)

        text = "Идёт генерация..."
        context.bot.send_message(chat_id=update.effective_chat.id, text= text )

        result = ST.transfering(
            path = WORK_DIR,
            chat_id = update.effective_chat.id,
            size = opt.size,
            num_steps=opt.num_steps,
            style_weight= opt.style_weight,
            content_weight = opt.content_weight
        )

        # Преобразует изображение в формат, который бот может отправить в Telegram.
        bio = BytesIO()
        bio.name = 'image.jpeg'
        result.save(bio, 'JPEG')
        bio.seek(0)
        context.bot.send_photo(chat_id=chat_id, photo=bio)

        # удаление рабочей папки пользователя если нужно
        shutil.rmtree(WORK_DIR/str(update.effective_chat.id))

def unknown(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Извини, я ограниченный бот и не понял команду" )

def main() -> None:
    updater = Updater(token=TOKEN)
    dispatcher = updater.dispatcher
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    start_handler = CommandHandler('start', start)

    # добавляем обработчики в `dispatcher
    dispatcher.add_handler(start_handler)

    photo_handler =  MessageHandler( Filters.photo, photo)
    dispatcher.add_handler(photo_handler)

    image_handler =  MessageHandler( Filters.document.image, image)
    dispatcher.add_handler(image_handler)

    transfer_style_handler = CommandHandler('transfer_style', transfer_style )
    dispatcher.add_handler(transfer_style_handler)

    unknown_handler = MessageHandler(Filters.text | Filters.document, unknown)
    dispatcher.add_handler(unknown_handler)

    updater.start_polling()
    print('Started')
    updater.idle() # работа до прекращения. нажать Ctrl+c
if __name__ == "__main__": # вызов функции main
    main()
