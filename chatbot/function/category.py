
from discord.ext import commands

import discord
import asyncio

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]


async def ranked_stores(message, bot, data, category, place="강남역"):
    stores = data.get_by_category(category)
    store_names = []
    store_stars = []

    if len(stores) >= 3:
        for res in stores[:3]:
            name, stars = res
            store_names.append(name)
            store_stars.append(stars)
    else:
        for res in stores:
            name, stars = res
            store_names.append(name)
            store_stars.append(stars)

    embed = discord.Embed(title="Show Top Stores",
                            description=f"{category} 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    
    for i in range(len(store_stars)):
        embed.add_field(name=emoji_list[i], value=f" {store_names[i]} {store_stars[i][9]}점 📝{store_stars[i][1]}건 😁 {store_stars[i][2]}% 🙂 {store_stars[i][3]}% 😫 {store_stars[i][4]}% ❗{store_stars[i][10]}건", inline=False)

    goback_emoji = ["◀", "⏪"]

    embed.add_field(name=goback_emoji[0], value="다른 카테고리 확인하기", inline=False)
    embed.add_field(name=goback_emoji[1], value="초기 메뉴로 돌아가기", inline=False)
    
    msg = await message.channel.send(embed=embed)

    for emoji in goback_emoji:
        await msg.add_reaction(emoji)

    def check_emoji(reaction, user):
        return str(reaction.emoji) in goback_emoji  and reaction.message.id == msg.id and user.bot == False
        
    try:
        reaction, user = await bot.wait_for(event='reaction_add', timeout=10.0, check=check_emoji)
        if reaction.emoji == goback_emoji[0]:
            return 1    # 다른 카테고리
        elif reaction.emoji == goback_emoji[1]:
            return 0    # 초기 메뉴
        
    except asyncio.TimeoutError:
        return -1

    # for emoji in emoji_list[:len(store_names)]:
    #     await msg.add_reaction(emoji)