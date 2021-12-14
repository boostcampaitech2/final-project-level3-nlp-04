
from discord.ext import commands
from discord.ext.commands import Bot

import discord
import asyncio

from function.review_gen import *
from function.category import *

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

async def func1(message, bot):
    embed = discord.Embed(title="FooReview Bot",
                          description="카테고리를 선택해주세요! 이모지를 눌러주세요",
                          color=0x00aaaa)

    embed.add_field(name="1️⃣", value="롯데리아-건대점, 불고기 버거 세트 1", inline=False)
    embed.add_field(name="2️⃣", value="피자왕비치킨공주 - 청주점, 불고기 피자 L", inline=False)
    embed.add_field(name="3️⃣", value="무국적식탁-광진점, 1인 우（牛）삼겹 스키야키 우동/1", inline=False)
    embed.add_field(name="4️⃣", value="직접 입력", inline=False)
    msg = await message.channel.send(embed=embed)
    for emoji in emoji_list[:4]:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False
    
    


async def func2(message, bot):
    categoryfunc = [show_category, chicken, serving, western, total]
    embed = discord.Embed(title="Choosing Category",
                            description="보고 싶은 카테고리를 이모지를 이용해 선택해주세요.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value="피자", inline=False)
    embed.add_field(name="2️⃣", value="치킨", inline=False)
    embed.add_field(name="3️⃣", value="1인분 주문", inline=False)
    embed.add_field(name="4️⃣", value="햄버거/양식", inline=False)
    embed.add_field(name="5️⃣", value="전체", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:5]:
        await msg.add_reaction(emoji)     

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False
        
    try:
        reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)
        if reaction.emoji in emoji_list:
            await categoryfunc[emoji_list.index(reaction.emoji)](reaction.message)
        
    except asyncio.TimeoutError:
        await message.channel.send('⚡ 20초가 지났습니다. 다시 !help를 입력해주세요.')
        return
    
async def func3(message, bot):
    pass

async def func4(message, bot):
    pass

async def func5(message, bot):
    pass