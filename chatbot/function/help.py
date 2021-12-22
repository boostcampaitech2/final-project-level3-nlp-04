
from discord.ext import commands
from discord.ext.commands import Bot
import io
import discord
import asyncio

from transformers.utils.dummy_pt_objects import DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST

from function.review import *
from function.category import *
from function.category_rank import RankReview

emoji_list = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


async def func1(message, bot):
    order = [
        ("하우마라탕-강남점", "1人 마라탕/2, 계란볶음밥/1"),
        ("달떡볶이-강남점", "초승달세트（떡볶이＋튀김1人＋순대1人＋음료1개）"),
        ("호야생과일쥬스&눈꽃빙수", "리얼생딸기눈꽃빙수/1, 청포도 생과일/1, 아이스 아메리카노/2"),
        ("직접 입력",),
    ]
    embed = discord.Embed(title="Review Generation",
                          description="리뷰 작성을 원하는 메뉴를 선택해주세요! 이모지를 눌러주세요",
                          color=0x00aaaa)

    for idx in range(len(order)):
        embed.add_field(name=emoji_list[idx], value=order[idx], inline=False)
    msg = await message.channel.send(embed=embed) # 다음 메세지 보여줌
    for emoji in emoji_list[:len(order)]:
        await msg.add_reaction(emoji) # 메세지에서 보여준 리스트 중 하나 선택하도록 해줌

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list[:len(order)] and reaction.message.id == msg.id and user.bot == False


    reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)
    if reaction.emoji in emoji_list:
        if emoji_list.index(reaction.emoji) == len(order) - 1:
            restaurant = await restaurant_enter(reaction.message, bot)
            menu = await menu_enter(reaction.message, bot)
        else:
            restaurant, menu = order[emoji_list.index(reaction.emoji)]

    food = await food_enter(reaction.message, bot)
    delvice = await delvice_enter(reaction.message, bot)
    review_text = await review_text_enter(reaction.message, bot,
                                          restaurant, menu,
                                          food, delvice)
    img = await image_enter(reaction.message, bot, menu + review_text)


    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list[:len(order)] and reaction.message.id == msg.id and user.bot == False

    # reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)

    embed = discord.Embed(title="Final Review",
                  description=f"{restaurant}의 {menu}, 음식 점수 {food}점 배달 및 서비스 점수 {delvice}점을 바탕으로 선택한 리뷰는",
                  color=0x00aaaa)

    embed.add_field(name="✔", value=f"{review_text}")
    msg = await message.channel.send(embed=embed)



    with io.BytesIO() as image_binary:
        img.save(image_binary, 'PNG')
        image_binary.seek(0)
        await message.channel.send(file=discord.File(fp=image_binary, filename='image.png'))
    return -1


async def func2(message, bot):
    
    embed = discord.Embed(title="Loading", description="가게별 랭킹 로딩 중입니다.........", color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    heart_emoji = ["❤","🧡","💛","💚","💙","💜","🤎","🖤","🤍"]
    for emoji in heart_emoji:
        await msg.add_reaction(emoji)

    RankedReview = RankReview(subway="강남역")

    while True:
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
                ret = await ranked_stores(reaction.message, bot, RankedReview, categorynames[emoji_list.index(reaction.emoji)])
                if ret == -1:
                    return -1
            
        except asyncio.TimeoutError:
            await message.channel.send('⚡ 20초가 지났습니다. 다시 !HELP를 입력해주세요.')
            return -1

        if ret == 0:
            break

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

    embed = discord.Embed(title="Finding by Keyword", description="검색하려는 키워드를 입력해주세요",  color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    message = await bot.wait_for(event='message')

    await message.channel.send(f'{message.content}를 검색하시는군요!')

    return -1

async def func4(message, bot):
    embed = discord.Embed(title="Keyword Input",
                          description="검색하고 싶은 키워드를 입력해주세요.",
                          color=0x00aaaa)
    await message.channel.send(embed=embed)
    message = await bot.wait_for(event='message')

    # message.content가 이제 입력받은 내용.

    # 불러오는 함수에 message.content 집어넣으면
    keyword = message.content

    # 돌려주고 추천해주는 함수에서 반환값으로 list를 주겠지?
    # 그럼 난 이 리스트에서 하나를 고르게 해줘야해

    tmpList = ["세진김밥", "세진떡볶이", "세진마라탕", "세진피자", "어깨치킨"]

    embed = discord.Embed(title="Recommended Restaurant",
                          description=f"입력하신 키워드 {keyword}에 기반하여 추천된 식당입니다.",
                          color=0x00aaaa)

    for idx in range(len(tmpList)):
        embed.add_field(name=emoji_list[idx], value=tmpList[idx], inline=False)
    msg = await message.channel.send(embed=embed) # 다음 메세지 보여줌
    for emoji in emoji_list[:len(tmpList)]:
        await msg.add_reaction(emoji) # 메세지에서 보여준 리스트 중 하나 선택하도록 해줌

    def check_emoji(reaction, user):
        return str(reaction.emoji) in emoji_list and reaction.message.id == msg.id and user.bot == False

    reaction, user = await bot.wait_for(event='reaction_add', timeout=20.0, check=check_emoji)  

    embed = discord.Embed(title="Selected Restaurant",
                            description=f"{keyword}의 대표 식당인 {tmpList[emoji_list.index(str(reaction.emoji))]}을(를) 선택하셨군요!\n좋은 선택입니다!",
                            color=0x00aaaa)
    msg = await message.channel.send(embed=embed)
    
    return -1
    
