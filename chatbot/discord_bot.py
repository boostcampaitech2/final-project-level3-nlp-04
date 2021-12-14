
import discord
import asyncio
from discord.ext import commands
from discord.ext.commands import Bot

token = 'OTE5ODc2MjA4Mzc4OTEyNzg4.YbcLeg.WqjPZPIj0qmtj_4y2iF59m-QV-Q' # 아까 메모해 둔 토큰을 입력합니다
client = discord.Client() # discord.Client() 같은 긴 단어 대신 client를 사용하겠다는 선언입니다.

@client.event
async def on_ready(): # 봇이 준비가 되면 1회 실행되는 부분입니다.
    await client.change_presence(status=discord.Status.online, activity=discord.Game("반갑습니다 :D"))
    print("I'm Ready!") # I'm Ready! 문구를 출력합니다.
    print(client.user.name) # 봇의 이름을 출력합니다.
    print(client.user.id) # 봇의 Discord 고유 ID를 출력합니다.

@client.event
async def on_message(message):
    if message.author.bot: # 채팅을 친 사람이 봇일 경우
        return None # 반응하지 않고 구문을 종료합니다.
    
    if message.content.startswith('!help'):
        embed = discord.Embed(title="FooReview Bot",
                              description="무엇을 도와드릴까요? 이모지를 눌러주세요",
                              color=0x00aaaa)
        embed.add_field(name="1️⃣", value="리뷰 생성", inline=False)
        embed.add_field(name="2️⃣", value="최근 1개월 BEST 음식점", inline=False)
        embed.add_field(name="3️⃣", value="키워드로 찾는 음식점", inline=False)
        embed.add_field(name="4️⃣", value="리뷰기반 추천 음식점", inline=False)
        msg = await message.channel.send(embed=embed)                
        await msg.add_reaction("1️⃣")
        await msg.add_reaction("2️⃣")
        await msg.add_reaction("3️⃣")
        await msg.add_reaction("4️⃣")
    
    if message.content == "!명령어":
        # 이 구문은 메시지가 보내진 채널에 메시지를 보내는 구문입니다.
        await message.channel.send("대답")
        # 이 아래 구문은 메시지를 보낸 사람의 DM으로 메시지를 보냅니다.
#         await message.author.send("응답")


@client.event
async def on_reaction_add(reaction, user):
    if user.bot == 1: #봇이면 패스
        return None
    
    if str(reaction.emoji) == "1️⃣":
        await reaction.message.channel.send(user.name + "님이 1번을 클릭")
    elif str(reaction.emoji) == "2️⃣":
        await reaction.message.channel.send(user.name + "님이 2번을 클릭")
    elif str(reaction.emoji) == "3️⃣":
        await reaction.message.channel.send(user.name + "님이 3번을 클릭")
    elif str(reaction.emoji) == "4️⃣":
        await reaction.message.channel.send(user.name + "님이 4번을 클릭")
 

client.run(token) # 아까 넣어놓은 토큰 가져다가 봇을 실행하라는 부분입니다
