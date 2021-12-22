import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import io
from discord.ext import commands
from discord.ext.commands import Bot
import discord
import asyncio
from elastic_img.retrieval_test import gen_img
from KoGPT2.gen_review import gen_rev
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
