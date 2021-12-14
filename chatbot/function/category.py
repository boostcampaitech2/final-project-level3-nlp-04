
from discord.ext import commands
from discord.ext.commands import Bot

import discord
import asyncio

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

async def show_category(message):
    embed = discord.Embed(title="Show Top Stores",
                            description="순위는 다음과 같습니다.",
                            color=0x00aaaa)
    # print(catnum)
    embed.add_field(name="1️⃣", value=" A가게 4.8점 📝1500건 😁 95% 🙂 3% 😫 2% ❗1건", inline=False)
    embed.add_field(name="2️⃣", value=" B가게 4.7점 📝2000건 😁 92% 🙂 5% 😫 2% ❗2건", inline=False)
    embed.add_field(name="3️⃣", value=" C가게 4.4점 📝500건 😁 88% 🙂 9% 😫 3% ❗3건", inline=False)

    msg = await message.channel.send(embed=embed)    
    for emoji in emoji_list[:3]:
        await msg.add_reaction(emoji)
        
        
async def pizza(message):
    pass

async def chicken(message):
    pass

async def serving(message):
    pass

async def western(message):
    pass

async def total(message):
    pass