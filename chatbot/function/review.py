from discord.ext import commands
from discord.ext.commands import Bot

import discord
import asyncio

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