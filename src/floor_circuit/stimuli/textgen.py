"""S1 语义完整性最小对文本生成：25 框架 × 12 填充 = 300 对/语言。

设计：
- 不完整前缀止于强投射点（EN：介词/及物动词/系词/冠词；ZH：介词/"把"/"是"/量词前）；
- 完整句 = 前缀 + 必要补足语（框架内嵌尾缀）；
- 同框架的 12 个条目配 12 个不同话头（opener），使 300 个不完整成员表面形式互不相同，
  而每对内部（完整 vs 不完整）共享同一话头，对比不被污染。
"""

from __future__ import annotations

# fmt: off
OPENERS_EN = [
    "Well,", "So,", "Actually,", "You know,", "Honestly,", "I mean,",
    "By the way,", "Anyway,", "Look,", "Okay so,", "Right,", "Listen,",
]
OPENERS_ZH = [
    "那个，", "就是，", "其实，", "我说，", "对了，", "话说，",
    "嗯，", "你看，", "这样，", "反正，", "后来，", "然后，",
]

# (不完整前缀, 补足语模板（含 {f}）, 12 个填充)
FRAMES_EN: list[tuple[str, str, list[str]]] = [
    ("I think we should move the meeting to", "{f}",
     ["Monday morning", "Tuesday afternoon", "Wednesday morning", "Thursday afternoon",
      "Friday morning", "next Monday", "next Friday", "the end of the month",
      "the tenth", "early next month", "the weekend", "five o'clock"]),
    ("She said the package was delivered to", "the {f} downstairs",
     ["office", "lobby", "mailroom", "front desk", "reception", "neighbor",
      "garage", "kitchen", "storage room", "back door", "security booth", "porch"]),
    ("The biggest problem with this plan is", "the {f}",
     ["budget", "timeline", "staffing", "paperwork", "location", "schedule",
      "pricing", "logistics", "wording", "funding", "deadline", "scope"]),
    ("After dinner we decided to", "{f}",
     ["go for a walk", "watch a movie", "play cards", "call it a night",
      "clean the kitchen", "take the dog out", "visit the neighbors",
      "get some ice cream", "sit on the porch", "do the dishes",
      "plan the trip", "play a board game"]),
    ("He couldn't make it to the party because", "his {f} was sick",
     ["son", "daughter", "wife", "mother", "father", "brother",
      "sister", "roommate", "grandmother", "grandfather", "uncle", "aunt"]),
    ("The recipe says you need two cups of", "{f}",
     ["flour", "sugar", "rice", "milk", "oats", "breadcrumbs", "chicken stock",
      "brown sugar", "shredded cheese", "chopped onions", "warm water", "coconut milk"]),
    ("I'm pretty sure I left my keys on", "the {f} by the door",
     ["table", "counter", "shelf", "bench", "dresser", "cabinet",
      "desk", "stool", "windowsill", "bookcase", "chair", "tray"]),
    ("They're planning to open a new", "{f} on Main Street",
     ["bakery", "bookstore", "pharmacy", "coffee shop", "hardware store", "gym",
      "diner", "flower shop", "barbershop", "grocery store", "pizza place", "laundromat"]),
    ("My favorite part of the trip was definitely", "the {f}",
     ["food", "beaches", "museums", "hiking", "architecture", "night market",
      "boat ride", "old town", "local music", "street food", "sunsets", "train ride"]),
    ("Could you please hand me the", "{f} on the shelf",
     ["scissors", "tape", "charger", "stapler", "notebook", "flashlight",
      "screwdriver", "remote", "folder", "glue", "batteries", "envelope"]),
    ("The doctor told me to stay away from", "{f} for a few weeks",
     ["coffee", "dairy", "sugar", "alcohol", "fried food", "red meat",
      "salt", "gluten", "spicy food", "soda", "shellfish", "caffeine"]),
    ("We completely ran out of", "{f} this morning",
     ["milk", "coffee", "eggs", "bread", "butter", "paper towels", "dish soap",
      "printer paper", "sugar", "cereal", "toothpaste", "laundry detergent"]),
    ("On the way home she stopped by", "the {f}",
     ["pharmacy", "bank", "post office", "bakery", "dry cleaner", "library",
      "supermarket", "hardware store", "gas station", "farmers market", "gym", "tailor"]),
    ("I've never really enjoyed", "{f}",
     ["camping", "jogging", "karaoke", "cooking shows", "horror movies", "board games",
      "long flights", "crowded malls", "roller coasters", "spicy food",
      "early mornings", "small talk"]),
    ("Before we sign anything, we need to talk to", "the {f}",
     ["lawyer", "landlord", "accountant", "contractor", "bank", "inspector",
      "insurance agent", "architect", "previous owner", "property manager",
      "realtor", "notary"]),
    ("The kids spent the whole afternoon playing in", "the {f}",
     ["backyard", "park", "basement", "pool", "garden", "tree house", "living room",
      "sandbox", "garage", "playground", "driveway", "attic"]),
    ("I finally got around to fixing", "the {f}",
     ["faucet", "fence", "printer", "garage door", "bike", "washing machine",
      "doorbell", "mailbox", "lamp", "cabinet hinge", "window screen", "vacuum"]),
    ("For the presentation tomorrow, don't forget to bring", "the {f}",
     ["slides", "projector", "handouts", "laptop", "adapter", "notes",
      "samples", "contracts", "name tags", "clicker", "badges", "prototype"]),
    ("Growing up, we used to spend every summer at", "my {f}'s place",
     ["grandmother", "uncle", "aunt", "cousin", "godmother", "grandfather",
      "stepbrother", "neighbor", "best friend", "great-aunt", "older sister",
      "family friend"]),
    ("If it keeps raining, we'll have to cancel", "the {f}",
     ["picnic", "barbecue", "parade", "game", "concert", "fireworks", "hike",
      "garage sale", "wedding rehearsal", "field trip", "race", "outdoor movie"]),
    ("The landlord promised to repair", "the {f} this week",
     ["heater", "roof", "elevator", "plumbing", "air conditioning", "front gate",
      "intercom", "water damage", "balcony railing", "garage door", "mailboxes",
      "staircase light"]),
    ("You should probably apologize to", "your {f}",
     ["sister", "coworker", "neighbor", "roommate", "teammate", "boss",
      "cousin", "classmate", "friend", "brother", "mentor", "assistant"]),
    ("Every morning the first thing she does is", "{f}",
     ["make coffee", "go for a jog", "stretch", "take a shower", "walk the dog",
      "check the news", "water the plants", "make breakfast", "meditate",
      "do a crossword", "listen to a podcast", "drink warm water"]),
    ("The whole argument started over", "a {f}",
     ["parking spot", "phone bill", "borrowed jacket", "missed call", "broken plate",
      "board game", "TV remote", "misunderstanding", "text message",
      "restaurant bill", "seating chart", "joke"]),
    ("Next weekend they're driving up to", "the {f}",
     ["mountains", "lake", "coast", "cabin", "vineyard", "city", "national park",
      "hot springs", "ski resort", "campground", "beach house", "waterfalls"]),
]

FRAMES_ZH: list[tuple[str, str, list[str]]] = [
    ("我觉得我们应该把会议改到", "{f}",
     ["周一上午", "周二下午", "周三上午", "周四下午", "周五上午", "下周一",
      "下周三", "月底", "十号", "下个月初", "周末", "五点以后"]),
    ("她说包裹已经送到了", "{f}",
     ["前台", "门卫室", "物业", "楼下超市", "快递柜", "办公室",
      "邻居家", "传达室", "收发室", "单元门口", "菜鸟驿站", "公司前台"]),
    ("这个方案最大的问题在于", "{f}",
     ["预算", "工期", "人手", "选址", "定价", "流程",
      "分工", "时间安排", "宣传", "物流", "合同条款", "售后"]),
    ("吃完饭我们打算去", "{f}",
     ["散步", "看电影", "逛超市", "打球", "唱歌", "喝茶",
      "江边走走", "商场转转", "接孩子", "健身房", "夜市", "朋友家坐坐"]),
    ("他没来是因为", "{f}",
     ["孩子发烧了", "临时加班", "车坏在路上了", "家里来客人了", "赶不上高铁",
      "身体不舒服", "出差还没回来", "搬家太累了", "手机没电联系不上",
      "堵在高速上了", "老家有事", "腰疼犯了"]),
    ("这个菜谱说需要两勺", "{f}",
     ["生抽", "老抽", "料酒", "白糖", "淀粉", "蚝油",
      "香油", "豆瓣酱", "米醋", "盐", "辣椒油", "芝麻酱"]),
    ("我好像把钥匙落在", "{f}了",
     ["办公室", "出租车上", "餐厅", "健身房", "我妈家", "前台",
      "会议室", "车里", "快递点", "理发店", "超市", "朋友家"]),
    ("他们准备在路口新开一家", "{f}",
     ["面馆", "奶茶店", "药店", "便利店", "理发店", "健身房",
      "火锅店", "书店", "花店", "烘焙店", "水果店", "咖啡馆"]),
    ("这次旅行我最喜欢的其实是", "{f}",
     ["当地的小吃", "海边的日落", "老城区", "夜市", "坐船那段", "爬山那天",
      "博物馆", "民宿的院子", "路上的风景", "街头音乐", "温泉", "集市"]),
    ("麻烦你把桌上的", "{f}递给我",
     ["剪刀", "充电器", "遥控器", "订书机", "笔记本", "胶带",
      "螺丝刀", "文件夹", "水杯", "眼镜", "钥匙", "数据线"]),
    ("医生让我这段时间少吃", "{f}",
     ["辣的", "油炸的", "甜食", "海鲜", "生冷的", "烧烤",
      "火锅", "咖啡", "浓茶", "腌制品", "夜宵", "冰的"]),
    ("家里今天早上正好用完了", "{f}",
     ["牛奶", "鸡蛋", "大米", "酱油", "洗衣液", "纸巾",
      "牙膏", "面粉", "食用油", "洗洁精", "咖啡豆", "垃圾袋"]),
    ("回家路上她顺便去了趟", "{f}",
     ["药店", "银行", "邮局", "菜市场", "干洗店", "图书馆",
      "超市", "水果店", "快递点", "理发店", "维修店", "裁缝店"]),
    ("我一直不太喜欢", "{f}",
     ["熬夜", "挤地铁", "开长会", "爬山", "吃香菜", "看恐怖片",
      "应酬", "逛街", "坐飞机", "排队", "早起", "唱K"]),
    ("签合同之前我们得先问问", "{f}",
     ["律师", "房东", "会计", "中介", "物业", "银行",
      "装修师傅", "保险公司", "原房主", "公司法务", "工程监理", "公证处"]),
    ("孩子们一下午都在", "{f}玩",
     ["后院", "公园", "地下室", "游泳池", "小区花园", "游乐场",
      "沙坑", "客厅", "车库", "操场", "树屋", "广场"]),
    ("我终于抽空修好了", "{f}",
     ["水龙头", "打印机", "车库门", "自行车", "洗衣机", "门铃",
      "台灯", "柜门", "纱窗", "吸尘器", "热水器", "插座"]),
    ("明天汇报别忘了带上", "{f}",
     ["幻灯片", "投影仪", "讲义", "笔记本电脑", "转接头", "样品",
      "合同", "翻页笔", "工牌", "原型机", "名单", "白板笔"]),
    ("小时候我们每年暑假都去", "{f}家住",
     ["外婆", "舅舅", "姑姑", "表哥", "爷爷", "奶奶",
      "姨妈", "叔叔", "堂姐", "邻居", "干妈", "老姨"]),
    ("要是一直下雨的话就只能取消", "{f}了",
     ["野餐", "烧烤", "运动会", "演唱会", "烟花", "徒步",
      "庙会", "球赛", "春游", "婚礼彩排", "马拉松", "露天电影"]),
    ("房东答应这周过来修", "{f}",
     ["暖气", "屋顶", "电梯", "下水道", "空调", "大门",
      "对讲机", "阳台栏杆", "车库门", "信箱", "楼道灯", "热水器"]),
    ("这事你最好当面跟", "{f}道个歉",
     ["你姐", "同事", "邻居", "室友", "队友", "老板",
      "表弟", "同学", "朋友", "你哥", "师傅", "助理"]),
    ("她每天早上起来的第一件事是", "{f}",
     ["冲咖啡", "晨跑", "做拉伸", "洗澡", "听播客", "遛狗",
      "做早餐", "浇花", "看新闻", "背单词", "冥想", "喝温水"]),
    ("他们俩吵架居然是因为", "{f}",
     ["一个车位", "话费账单", "一件外套", "一个未接电话", "一个盘子", "桌游规则",
      "电视遥控器", "一条微信", "聚餐买单", "座位安排", "一个玩笑", "遛狗路线"]),
    ("下周末他们打算自驾去", "{f}",
     ["山里", "湖边", "海边", "古镇", "露营基地", "温泉",
      "滑雪场", "葡萄园", "国家公园", "郊区民宿", "沙滩", "瀑布"]),
]
# fmt: on


def _join_opener_en(opener: str, sentence: str) -> str:
    head = sentence[0]
    body = sentence if sentence.startswith(("I ", "I'")) else head.lower() + sentence[1:]
    return f"{opener} {body}"


def generate_s1(lang: str) -> list[dict]:
    """返回 300 条：{id, lang, frame, complete, incomplete}。"""
    if lang == "en":
        frames, openers = FRAMES_EN, OPENERS_EN
    elif lang == "zh":
        frames, openers = FRAMES_ZH, OPENERS_ZH
    else:
        raise ValueError(f"未知语言 {lang}")
    items = []
    for fi, (prefix, completion_tpl, fillers) in enumerate(frames):
        if len(fillers) != 12:
            raise ValueError(f"{lang} 框架 {fi} 的填充数应为 12，实际 {len(fillers)}")
        for gi, (filler, opener) in enumerate(zip(fillers, openers, strict=True)):
            completion = completion_tpl.format(f=filler)
            if lang == "en":
                complete = _join_opener_en(opener, f"{prefix} {completion}.")
                incomplete = _join_opener_en(opener, prefix)
            else:
                complete = f"{opener}{prefix}{completion}。"
                incomplete = f"{opener}{prefix}"
            items.append(
                {
                    "id": f"s1_{lang}_{fi:02d}_{gi:02d}",
                    "lang": lang,
                    "frame": fi,
                    "complete": complete,
                    "incomplete": incomplete,
                }
            )
    completes = [it["complete"] for it in items]
    if len(set(completes)) != len(completes):
        raise ValueError("S1 完整句存在重复")
    if len(items) != 25 * 12:
        raise ValueError(f"S1 数量应为 300，实际 {len(items)}")
    return items


def s2_texts(stimuli_cfg: dict) -> dict:
    """S2 各类打断文本（指令性打断 / backchannel；跨语言版即互换语言喂给对方模型）。"""
    s2 = stimuli_cfg["s2"]
    return {
        "instruct": dict(s2["interrupt_text"]),
        "backchannel": dict(s2["backchannel_text"]),
    }
