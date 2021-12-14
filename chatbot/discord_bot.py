
import discord
import asyncio
from discord.ext import commands
from discord.ext.commands import Bot
from discord.ext.commands.core import check

token = 'OTE5ODgyODc3ODIxMzk5MDQw.YbcRsA.Fx2r1ivN-6ZYmKY8IEI1n523rH4' # 아까 메모해 둔 토큰을 입력합니다
client = discord.Client() # discord.Client() 같은 긴 단어 대신 client를 사용하겠다는 선언입니다.
emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

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
                              description="안녕하세요, 무엇을 도와드릴까요? 이모지를 눌러주세요",
                              color=0x00aaaa)
        embed.add_field(name="1️⃣", value="리뷰 생성", inline=False)
        embed.add_field(name="2️⃣", value="최근 1개월 BEST 음식점", inline=False)
        embed.add_field(name="3️⃣", value="키워드로 찾는 음식점", inline=False)
        embed.add_field(name="4️⃣", value="리뷰기반 추천 음식점", inline=False)
        msg = await message.channel.send(embed=embed)   

        for emoji in emoji_list[:4]:
            await msg.add_reaction(emoji)

        def check_emoji(reaction, user):
            return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

        try:
            reaction, user = await client.wait_for(event='reaction_add', timeout=60.0, check=check_emoji)
            idx = emoji_list.index(str(reaction.emoji)) + 1
            if idx == 2:  # 두 번째 메뉴 진입 위함
                await choose_category(message)
            pass
        except asyncio.TimeoutError:
            return

    
    if message.content == "!명령어":
        # 이 구문은 메시지가 보내진 채널에 메시지를 보내는 구문입니다.
        await message.channel.send("대답")
        # 이 아래 구문은 메시지를 보낸 사람의 DM으로 메시지를 보냅니다.
#         await message.author.send("응답")



'''
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
'''

async def choose_category(message):    
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
        reaction, user = await client.wait_for(event='reaction_add', timeout=60.0, check=check_emoji)
        idx = emoji_list.index(str(reaction.emoji)) + 1
        if idx == 3:  # 세 번째 메뉴 진입 위함
            await show_category(message)
        pass
    except asyncio.TimeoutError:
        return       
    
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

client.run(token) # 아까 넣어놓은 토큰 가져다가 봇을 실행하라는 부분입니다