import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import io
from discord.ext import commands
from discord.ext.commands import Bot
import discord
import asyncio
from chatbot.model.elastic_img.retrieval_test import gen_img
from chatbot.model.kogpt.gen_review import gen_rev
emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]


async def menu_enter(message, bot):
    embed = discord.Embed(title="FooReview Bot",
                          description="메뉴를 입력해주세요",
                          color=0x00aaaa)
    await message.channel.send(embed=embed)
    message = await bot.wait_for(event='message')

    return message.content


async def restaurant_enter(message, bot):
    embed = discord.Embed(title="FooReview Bot",
                          description="음식점을 입력해주세요",
                          color=0x00aaaa)
    await message.channel.send(embed=embed)
    message = await bot.wait_for(event='message')

    return message.content

async def review_text_enter(message, bot, restaurant, menu, food, delvice):
    embed = discord.Embed(title="Review Generated",
                          description=f"생성된 리뷰입니다. 하나를 선택하세요.",
                          color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    reviews = gen_rev(restaurant, menu, food, delvice)

    for r_idx in range(len(reviews)):
        embed.add_field(name=emoji_list[r_idx], value=reviews[r_idx], inline=False)
    msg = await message.channel.send(embed=embed)
    for emoji in emoji_list[:len(reviews)]:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False
    reaction, user = await bot.wait_for(event='reaction_add', check=check_emoji)
    if reaction.emoji in emoji_list:
        return reviews[emoji_list.index(reaction.emoji)]


async def image_enter(message, bot, review):

    text = review
    pil_img = gen_img(text)

    embed = discord.Embed(title="FooReview Bot",
                          description="리뷰 사진을 골라주세요",
                          color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    for img in pil_img:
        with io.BytesIO() as image_binary:
            img.save(image_binary, 'PNG')
            image_binary.seek(0)
            await message.channel.send(file=discord.File(fp=image_binary, filename='image.png'))
    for emoji in emoji_list[:4]:
        await msg.add_reaction(emoji) # 번호 리스트 제공

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False
    reaction, user = await bot.wait_for(event='reaction_add', check=check_emoji)
    if reaction.emoji in emoji_list:
        return pil_img[emoji_list.index(reaction.emoji)]


async def food_enter(message, bot):
    embed = discord.Embed(title="FooReview Bot",
                          description="음식 별점을 입력해주세요",
                          color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    for emoji in [emoji_list[0], emoji_list[2], emoji_list[4]]:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

    reaction, user = await bot.wait_for(event='reaction_add', check=check_emoji)
    if reaction.emoji in emoji_list:
        return emoji_list.index(reaction.emoji)+1

async def delvice_enter(message, bot):
    embed = discord.Embed(title="FooReview Bot",
                          description="배달 및 서비스 별점을 입력해주세요",
                          color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    for emoji in [emoji_list[0], emoji_list[2], emoji_list[4]]:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

    reaction, user = await bot.wait_for(event='reaction_add', check=check_emoji)
    if reaction.emoji in emoji_list:
        return emoji_list.index(reaction.emoji) + 1


def review_gen(food, delvice):
    # 앞이 food, 뒤가 delvice

    rev55 = ["맛있게 잘 먹었어용! 서비스도 원하는 토핑으로 잘 받았습니당!",
            "오늘도 튀김이랑 국물떡볶이 따뜻하게 잘먹었습니다~ㅎ 강추!!",
            "두번째 시켜먹는곳인데 너무 맛있습니당 이제 떡볶이는 여기서만 시켜야겠어요! 배달도 빨랐어요~"]
    rev53 = ["양도 많고 맛있었습니다!",
            "맛있어요 누가시켜도 후회하지 않을  맛",
            "오자마자 바로 먹어서 사진을 못찍었네요ㅠ  잘먹었습니당~~ㅎㅎ"]
    rev51 = ["음식과 양은 정말 만족했는데 배달이 너무 오래걸려서 아쉬웠습니다..",
            "배달 예상시간보다 늦게왔고 튀김이 식은 정도가 아니라 차가웠습니다. 떡볶이는 맛있어요",
            "언제 먹어도 맛있는 맛이네요 다만 타지점에 비해 오래걸리는 것 같아서 아쉬워요"]

    rev35 = ["배달도 빠르고 양도 많은데 맛이 뭔가 좀 변했어요. 원래 좋아하던 프렌차이즈였는데 뭔가 딱딱해지고 느끼해졌습니다ㅜ",
            "생각보다 좀 맛은 밋밋했어요. 양은 다른곳과비슷하고 배달도 빨랐어요",
            "점심시간이었는데도 배달되게 빨랐어요 ㄱㅅㄱㅅ"]

    rev33 = ["배달는보통입니다  안전운전하세요",
            "맛있는데 치즈가 좀 느끼해요 치토스 맛이 난다고 해야하나",
            "괜찮아요 나쁘지 않아요"]

    rev31 = ["맛있는데 느끼해요 오뎅튀김 안왔음 ㅠㅠ",
            "배달이 넘 늦어요~~ 떡볶이가 따뜻하지안아영 배달이 늦었어 그런가  ㅎㅎ",
            "하하  또 차게식어왔네요..ㅎ 겨울엔 얼어서 오겠어요..ㅎ"]

    rev15 = ["배달 30분이나 빨리 오셨어요! 그리고 양도 많이 주셨습니다 근데 맛이 심각하게 없어요...",
            "배달빠르구 먹을만한데..양이너무적구 떡볶이가 달아용..",
            "안매운맛 주문인데도 맵네요.. 튀김은 너무 튀겨짐.. 배달 겁나 빨라요"]

    rev13 = ["너무..맛이없다..",
            "리뷰보고 시킴 .가격의비해 가성비 비추임",
            "저번 주문은 맛있게 먹었는데 이번은 별로네요"]

    rev11 = ["맛있어서 여러 번 시켜 먹었는데 점점 양도 줄고 배달도 느려지네요 처음 시켰을 때에 비해서 양이 반으로 줄었네요 좁은데 꾹꾹 담으니까 떡볶이도 눌러붙어서 별로고요",
            "배달시간보다 10분늦게오고 음식 다식고 순대 내장 없음",
            "매운맛 시켰는데. 주문실수로 너무 밍밍한게 왔어요. 나중에 매운소스와 음료 따로 보내주긴 했는데. 이미 다 식었어요."]

    reviews = [rev55, rev53, rev51, rev35, rev33, rev31, rev15, rev13, rev11]

    food = 2 - (food-1)//2
    delvice = 2 - (delvice-1)//2

    return reviews[food*3+delvice]