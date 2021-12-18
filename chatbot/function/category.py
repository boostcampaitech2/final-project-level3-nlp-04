
from discord.ext import commands
from discord.ext.commands import Bot

import discord
import asyncio
from function.category_rank import RankReview

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

async def rank_reviews(message, category, place="강남역"):
    RankedReview = RankReview(place)
    await ranked_stores(message, RankedReview, category)

async def ranked_stores(message, data, category):
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
                            description=f"기다려주셔서 감사합니다! {category} 순위는 다음과 같습니다.",
                            color=0x00aaaa)
    
    for i in range(len(store_stars)):
        embed.add_field(name=emoji_list[i], value=f" {store_names[i]} {store_stars[i][0]}점 📝{store_stars[i][1]}건 😁 {store_stars[i][2]}% 🙂 {store_stars[i][3]}% 😫 {store_stars[i][4]}% ❗1건", inline=False)
    
    msg = await message.channel.send(embed=embed)

    # for emoji in emoji_list[:len(store_names)]:
    #     await msg.add_reaction(emoji)