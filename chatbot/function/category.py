
from discord.ext import commands
from discord.ext.commands import Bot

import discord
import asyncio
from function.rank_review import get_ranked_stores, get_by_category

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]      


async def chicken(message):
    stores = get_by_category("치킨", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 치킨 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)
        
async def pizza_western(message):
    stores = get_by_category("피자/양식", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 피자/양식 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)


async def chinese(message):
    stores = get_by_category("중국집", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 중국집 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)

async def korean(message):
    stores = get_by_category("한식", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 한식 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)


async def japanese(message):
    stores = get_by_category("일식/돈까스", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 일식/돈까스 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)


async def pigs(message):
    stores = get_by_category("족발/보쌈", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 족발/보쌈 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)


async def midnight_food(message):
    stores = get_by_category("야식", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 야식 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)
    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)


async def snack(message):
    stores = get_by_category("분식", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 분식 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)


async def cafe_desserts(message):
    stores = get_by_category("카페/디저트", "강남역")
    store_names = []
    store_stars = []
    for res in stores[:3]:
        name, stars = res
        store_names.append(name)
        store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description="기다려주셔서 감사합니다! 카페/디저트 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    embed.add_field(name="1️⃣", value=f" {store_names[0]} {store_stars[0][0]}점 📝{store_stars[0][1]}건 😁 {store_stars[0][2]}% 🙂 {store_stars[0][3]}% 😫 {store_stars[0][4]}% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=f" {store_names[1]} {store_stars[1][0]}점 📝{store_stars[1][1]}건 😁 {store_stars[1][2]}% 🙂 {store_stars[1][3]}% 😫 {store_stars[1][4]}% ❗1건", inline=False)
    embed.add_field(name="3️⃣", value=f" {store_names[2]} {store_stars[2][0]}점 📝{store_stars[2][1]}건 😁 {store_stars[2][2]}% 🙂 {store_stars[2][3]}% 😫 {store_stars[2][4]}% ❗1건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)