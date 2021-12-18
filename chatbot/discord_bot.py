
from discord.ext import commands
from discord.ext.commands import Bot

import discord
import asyncio

from function.help import *

token = 'OTE5ODgyODc3ODIxMzk5MDQw.YbcRsA.Fx2r1ivN-6ZYmKY8IEI1n523rH4' # 아까 메모해 둔 토큰을 입력합니다
bot = commands.Bot(command_prefix='!')

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "💤"]
helpfunc = [func1, func2, func3, func4, func5]
        
@bot.event
async def on_ready(): # 봇이 준비가 되면 1회 실행되는 부분입니다.
    await bot.change_presence(status=discord.Status.online, activity=discord.Game("반갑습니다 :D"))
    print("I'm Ready!") # I'm Ready! 문구를 출력합니다.
    print(bot.user.name) # 봇의 이름을 출력합니다.
    print(bot.user.id) # 봇의 Discord 고유 ID를 출력합니다.

@bot.command(name='HELP')
async def help(message):
    while True:
        embed = discord.Embed(title="FooReview Bot",
                            description="무엇을 도와드릴까요? 이모지를 눌러주세요",
                            color=0x00aaaa)
        embed.add_field(name="1️⃣", value="리뷰 생성", inline=False)
        embed.add_field(name="2️⃣", value="최근 1개월 BEST 음식점", inline=False)
        embed.add_field(name="3️⃣", value="키워드로 찾는 음식점", inline=False)
        embed.add_field(name="4️⃣", value="리뷰기반 추천 음식점", inline=False)
        embed.add_field(name="💤", value="프로그램 종료하기", inline=False)
        msg = await message.channel.send(embed=embed)

        for emoji in emoji_list[:4]:
            await msg.add_reaction(emoji)
        await msg.add_reaction("💤")

        def check_emoji(reaction, user):
            return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False
            
        try:
            reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)
            if reaction.emoji in emoji_list[:4]:
                ret = await helpfunc[emoji_list.index(reaction.emoji)](reaction.message, bot)
            if reaction.emoji == "💤" or ret == -1:
                await message.channel.send("🎈 이용해주셔서 감사합니다. 프로그램을 종료합니다. ")
                return
            
        except asyncio.TimeoutError:
            print("error")
            await message.channel.send('⚡ 20초가 지났습니다. 다시 !HELP를 입력해주세요.')
            return



bot.run(token) # 아까 넣어놓은 토큰 가져다가 봇을 실행하라는 부분입니다
