
from discord.ext import commands
from discord.ext.commands import Bot
import io
import discord
import asyncio

from function.review import *
from function.category import *

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]


async def func1(message, bot):
    order = [
        ("롯데리아-건대점", "불고기 버거 세트 1"),
        ("피자왕비치킨공주 - 청주점", "불고기 피자 L"),
        ("무국적식탁-광진점", "1인 우（牛）삼겹 스키야키 우동/1"),
        ("직접 입력",),
    ]
    embed = discord.Embed(title="FooReview Bot",
                          description="카테고리를 선택해주세요! 이모지를 눌러주세요",
                          color=0x00aaaa)

    for idx in range(len(order)):
        embed.add_field(name=emoji_list[idx], value=order[idx], inline=False)
    msg = await message.channel.send(embed=embed) # 다음 메세지 보여줌
    for emoji in emoji_list[:len(order)]:
        await msg.add_reaction(emoji) # 메세지에서 보여준 리스트 중 하나 선택하도록 해줌

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

    try:
        reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)
        if reaction.emoji in emoji_list:
            if emoji_list.index(reaction.emoji) == len(order) - 1:
                restaurant = await restaurant_enter(reaction.message, bot)
                menu = await menu_enter(reaction.message, bot)
            else:
                restaurant, menu = order[emoji_list.index(reaction.emoji)]

        food = await food_enter(reaction.message, bot)
        delvice = await delvice_enter(reaction.message, bot)
        img = await image_enter(reaction.message, bot, "치킨이 다 식어서 왔어요")

        await message.channel.send(f"음식점은 {restaurant}, 메뉴는 {menu}, 음식 점수는 {food}점, 배달 및 서비스 점수는 {delvice}점")
        await message.channel.send("훌륭한 사진이군요^^")
        with io.BytesIO() as image_binary:
            img.save(image_binary, 'PNG')
            image_binary.seek(0)
            await message.channel.send(file=discord.File(fp=image_binary, filename='image.png'))
    except asyncio.TimeoutError:
        await message.channel.send('⚡ 20초가 지났습니다. 다시 !help를 입력해주세요.')
        return


async def func2(message, bot):
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
        reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)
        if reaction.emoji in emoji_list:
            await rank_reviews(reaction.message, categorynames[emoji_list.index(reaction.emoji)])
        
    except asyncio.TimeoutError:
        await message.channel.send('⚡ 20초가 지났습니다. 다시 !help를 입력해주세요.')
        return
    
async def func3(message, bot):
    order = [
        ("롯데리아-건대점", "불고기 버거 세트 1"),
        ("피자왕비치킨공주 - 청주점", "불고기 피자 L"),
        ("무국적식탁-광진점", "1인 우（牛）삼겹 스키야키 우동/1"),
        ("직접 입력",),
    ]
    embed = discord.Embed(title="FooReview Bot",
                          description="카테고리를 선택해주세요! 이모지를 눌러주세요",
                          color=0x00aaaa)

    for idx in range(len(order)):
        embed.add_field(name=emoji_list[idx], value=order[idx], inline=False)
    msg = await message.channel.send(embed=embed)
    for emoji in emoji_list[:len(order)]:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

    try:
        reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)
        if reaction.emoji in emoji_list:
            if emoji_list.index(reaction.emoji) == len(order) - 1:
                restaurant = await restaurant_enter(reaction.message, bot)
                menu = await menu_enter(reaction.message, bot)
            else:
                restaurant, menu = order[emoji_list.index(reaction.emoji)]

        food = await food_enter(reaction.message, bot)
        delvice = await delvice_enter(reaction.message, bot)

        await message.channel.send(f"음식점은 {restaurant}, 메뉴는 {menu}, 음식 점수는 {food}점, 배달 및 서비스 점수는 {delvice}점")

    except asyncio.TimeoutError:
        await message.channel.send('⚡ 20초가 지났습니다. 다시 !help를 입력해주세요.')
        return


async def func4(message, bot):
    pass

async def func5(message, bot):
    pass