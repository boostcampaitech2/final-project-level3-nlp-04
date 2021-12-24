import os
import sys

import pandas as pd
from psutil import disk_io_counters
import torch

from retriever.utils import Config, get_encoders, get_path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset
from discord.ext import commands
from discord.ext.commands import Bot
import io
import discord
import asyncio

from transformers.utils.dummy_pt_objects import DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST

from chatbot.function.review import *
from chatbot.function.category import *
from chatbot.function.category_rank import RankReview
from chatbot.function.recommend import RecommendRestaurant
from chatbot.function.style_transfer import review_transfer

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


async def func1(message, bot):
    dataset = load_dataset('samgin/FooReview')['train']
    df = pd.DataFrame()
    df['restaurant'] = dataset['restaurant']
    df['menu'] = dataset['menu']


    order = df.sample(3).values.tolist()
    order.append(["직접 입력",])

    embed = discord.Embed(title="Review Generation",
                          description="리뷰 작성을 원하는 메뉴를 선택해주세요! 이모지를 눌러주세요",
                          color=0x00aaaa)

    for idx in range(len(order)):
        embed.add_field(name=emoji_list[idx], value=order[idx], inline=False)
    msg = await message.channel.send(embed=embed) # 다음 메세지 보여줌
    for emoji in emoji_list[:len(order)]:
        await msg.add_reaction(emoji) # 메세지에서 보여준 리스트 중 하나 선택하도록 해줌

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list[:len(order)] and reaction.message.id == msg.id and user.bot == False


    reaction, user = await bot.wait_for(event='reaction_add', timeout=60.0, check=check_emoji)
    if reaction.emoji in emoji_list:
        if emoji_list.index(reaction.emoji) == len(order) - 1:
            restaurant = await restaurant_enter(reaction.message, bot)
            menu = await menu_enter(reaction.message, bot)
        else:
            restaurant, menu = order[emoji_list.index(reaction.emoji)]

    food = await food_enter(reaction.message, bot)
    delvice = await delvice_enter(reaction.message, bot)
    review_text = await review_text_enter(reaction.message, bot,
                                          restaurant, menu,
                                          food, delvice)

    embed = discord.Embed(title="Image Generation",
                          description=f"{review_text}의 이미지를 생성할까요?",
                          color=0x00aaaa)

    msg = await message.channel.send(embed=embed)
    ox_emoji_list = ["⭕", "❌"]
    for emoji in ox_emoji_list:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in ox_emoji_list and reaction.message.id == msg.id and user.bot == False

    reaction, user = await bot.wait_for(event='reaction_add', timeout=60.0, check=check_emoji)

    flag = False
    if reaction.emoji == "⭕":
        flag = True
        img = await image_enter(reaction.message, bot, f'{restaurant} {menu} {review_text}')

    embed = discord.Embed(title="Review Style Transfer",
                description=f"{review_text}의 style transfer된 리뷰를 확인할까요?",
                color=0x00aaaa)

    msg = await message.channel.send(embed=embed)
    for emoji in ox_emoji_list:
        await msg.add_reaction(emoji) 

    reaction, user = await bot.wait_for(event='reaction_add', timeout=60.0, check=check_emoji)
    if reaction.emoji == "⭕":
        transferred_review = review_transfer(review_text)
        review_text = transferred_review

    # reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)

    embed = discord.Embed(title="Final Review",
                  description=f"{restaurant}의 {menu}, 음식 점수 {food}점 배달 및 서비스 점수 {delvice}점을 바탕으로 선택한 리뷰는",
                  color=0x00aaaa)

    embed.add_field(name="✔", value=f"{review_text}")
    msg = await message.channel.send(embed=embed)

    if flag:
        with io.BytesIO() as image_binary:
            img.save(image_binary, 'PNG')
            image_binary.seek(0)
            await message.channel.send(file=discord.File(fp=image_binary, filename='image.png'))
    return -1


async def func2(message, bot):
    
    embed = discord.Embed(title="Loading", description="가게별 랭킹 로딩 중입니다.........", color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    heart_emoji = ["❤","🧡","💛","💚","💙","💜","🤎","🖤","🤍"]
    for emoji in heart_emoji:
        await msg.add_reaction(emoji)

    RankedReview = RankReview(subway="강남역")

    while True:
        categorynames = ['치킨', '피자/양식', '중국집', '한식', '일식/돈까스', '족발/보쌈', '야식', '분식', '카페/디저트']  
        embed = discord.Embed(title="Choosing Category",
                                description="보고 싶은 카테고리를 이모지를 이용해 선택해주세요.",
                                color=0x00aaaa)
        
        for i in range(len(categorynames)):
            embed.add_field(name=emoji_list[i], value=categorynames[i], inline=False)

        msg = await message.channel.send(embed=embed)    
        for emoji in emoji_list[:len(categorynames)]:
            await msg.add_reaction(emoji)     

        def check_emoji(reaction, user):
            return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False
            
        try:
            reaction, user = await bot.wait_for(event='reaction_add', timeout=60.0, check=check_emoji)
            if reaction.emoji in emoji_list:
                ret = await ranked_stores(reaction.message, bot, RankedReview, categorynames[emoji_list.index(reaction.emoji)])
                if ret == -1:
                    return -1
            
        except asyncio.TimeoutError:
            await message.channel.send('⚡ 60초가 지났습니다. 다시 !HELP를 입력해주세요.')
            return -1

        if ret == 0:
            break

async def func3(message, bot):
    path_dict = get_path()
    config = Config().get_config(os.path.join(path_dict['configs_path'], 'klue_bert_base_model.yaml'))

    tokenizer, p_encoder, q_encoder = get_encoders(config)
    p_encoder.load_state_dict(torch.load(os.path.join(path_dict['output_path'], 'p_encoder', f'{config.run_name}.pt')))
    q_encoder.load_state_dict(torch.load(os.path.join(path_dict['output_path'], 'q_encoder', f'{config.run_name}.pt')))

    recommend_restaurant = RecommendRestaurant(config, tokenizer, p_encoder, q_encoder, path_dict['data_path'])

    while True:
        embed = discord.Embed(title="Keyword Input",
                              description="검색하고 싶은 키워드를 입력해주세요.\n종료하려면 exit 를 입력해주세요.",
                              color=0x00aaaa)
        await message.channel.send(embed=embed)
        message = await bot.wait_for(event='message')

        if message.content == 'exit':
            break

        # message.content가 이제 입력받은 내용.

        # 불러오는 함수에 message.content 집어넣으면
        keyword = [' '.join(['#' + keyword for keyword in message.content.split()])]

        # 돌려주고 추천해주는 함수에서 반환값으로 list를 주겠지?
        # 그럼 난 이 리스트에서 하나를 고르게 해줘야해

        response = recommend_restaurant.get_restaurant(keyword)
        restaurant_list = response['top_10_restaurant']
        address_list = response['top_10_address']

        embed = discord.Embed(title="Recommended Restaurant",
                              description=f"입력하신 키워드 {keyword}에 기반하여 추천된 식당입니다.",
                              color=0x00aaaa)

        for idx in range(10):
            embed.add_field(name=emoji_list[idx], value=restaurant_list[idx], inline=False)
        msg = await message.channel.send(embed=embed) # 다음 메세지 보여줌
        for emoji in emoji_list[:len(restaurant_list)]:
            await msg.add_reaction(emoji) # 메세지에서 보여준 리스트 중 하나 선택하도록 해줌

        def check_emoji(reaction, user):
            return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

        reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)

        embed = discord.Embed(title="Selected Restaurant",
                                description=f"{keyword}의 대표 식당인 {restaurant_list[emoji_list.index(str(reaction.emoji))]}을(를) 선택하셨군요!\n네이버에 {address_list[emoji_list.index(str(reaction.emoji))]} {restaurant_list[emoji_list.index(str(reaction.emoji))]} 를 검색하세요!",
                                color=0x00aaaa)
        msg = await message.channel.send(embed=embed)
    
    return -1
    
