"""材料经济：库存、配方、工具与公共设施（"世界规则"的执行者）。

原始模块只硬编码了一条配方（2 木 + 1 石 → 石斧）和一个公共地点
（屏幕中央的社区）。本版本保持这些旧入口逐字节兼容（``craft`` /
``contribute`` 的行为与原来完全一致），并在其上加一层**数据驱动**：

* :class:`Inventory` 现在携带 ``fiber``/``cord``/``basket`` 与负重
  上限（背篓提高上限），"更多技术"真实改变移动与采集经济；
* :meth:`Technology.execute_recipe` 是把材料变成技术的**唯一**合法
  途径。无论谁请求（规划器、教师还是知识实验），它都照常扣投入、
  发产出——世界模拟器保持唯一权威，规划器无法"宣告"出新工具；
* :meth:`Technology.simulate_recipe` 是 :mod:`newworld.knowledge`
  用的验证路径：一次尝试可能*失败*（``recipe.difficulty``）且仍然
  消耗材料，这让"发现"成为真实的实验而非脚本解锁；
* :meth:`Technology.apply_facility_effects` 让建成的设施产生真实、
  可测量的效果（取暖、休息恢复、取水、制作可靠性）。

配方用鸭子类型（``recipe_id``、``inputs``、``outputs``、
``requires_tool``、``difficulty``……），因此本模块**不** import
:mod:`newworld.knowledge`（避免循环依赖）。目录放在
``knowledge.RECIPE_CATALOG``。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import materials

#: 会进入原料库存的产出
INVENTORY_OUTPUTS = frozenset({"wood", "stone", "fiber", "cord"})
#: 属于随身装备的产出
TOOL_OUTPUTS = frozenset({"axe", "pickaxe", "basket", "pot"})
#: 属于公共建筑的产出（由文明层创建）
FACILITY_OUTPUTS = frozenset({"campfire", "hut", "well", "workshop", "granary"})

#: 设施类别 -> (生效半径 px, 简短中文名)
FACILITY_RADIUS = {
    "campfire": 160.0,
    # 粮仓养活住在它周围的每一个人：第一项让群体超出浆果丛
    # 自然承载力的技术
    "granary": 240.0,
    "hut": 200.0,
    "well": 90.0,
    "workshop": 160.0,
}

#: 有限组合式创新原型（newworld/innovation.py）产生、由本模块应用的效果键
CUSTOM_EFFECT_KEYS = frozenset(
    {"harvest_bonus", "capacity_bonus", "energy_regen", "water_regen", "craft_success_bonus"}
)


@dataclass
class Inventory:
    """居民随身携带的东西。

    ``capacity()`` 限制原料数量（``wood/stone/fiber/cord``）；装备与
    工具不计入。背篓（技术）提高上限——这是第一个"发现的技术改变
    未来行为"而不仅是屏幕上多个数字的机制。
    """

    wood: int = 0
    stone: int = 0
    fiber: int = 0
    cord: int = 0
    tool: str | None = None
    basket: int = 0
    capacity_base: int = 8
    #: 通过验证的组合式创新带来的额外负重（真实效果，
    #: 经 ``Technology.register_custom_effect`` 登记）
    capacity_bonus: int = 0

    # ------------------------------------------------------------------
    def capacity(self) -> int:
        """负重上限（基础 + 每个背篓 +4 + 创新加成）。"""
        return self.capacity_base + 4 * self.basket + int(self.capacity_bonus)

    def load(self) -> int:
        """当前原料总重量。"""
        return self.wood + self.stone + self.fiber + self.cord

    def free(self) -> int:
        """剩余可携带量。"""
        return max(0, self.capacity() - self.load())

    def add(self, kind: str, amount: int = 1) -> bool:
        """有空间就加入原料；返回是否装得下。"""
        if kind not in INVENTORY_OUTPUTS:
            return False
        amount = int(amount)
        if amount <= 0:
            return True
        if self.load() + amount > self.capacity():
            return False
        setattr(self, kind, int(getattr(self, kind)) + amount)
        return True

    def take(self, kind: str, amount: int = 1) -> bool:
        """原料足够就扣除；返回是否真的扣了。"""
        if kind not in INVENTORY_OUTPUTS:
            return False
        current = int(getattr(self, kind, 0))
        if current < int(amount):
            return False
        setattr(self, kind, current - int(amount))
        return True

    def has(self, kind: str, amount: int = 1) -> bool:
        """是否持有至少 amount 个某原料。"""
        return int(getattr(self, kind, 0)) >= int(amount)

    def to_dict(self) -> dict[str, object]:
        """序列化为普通字典（telemetry / 存档用）。"""
        return {
            "wood": self.wood,
            "stone": self.stone,
            "fiber": self.fiber,
            "cord": self.cord,
            "tool": self.tool,
            "basket": self.basket,
            "capacity": self.capacity(),
        }


@dataclass
class CommunitySite:
    """旧版公共地点（为兼容与测试保留；新设施走 CivilizationState）。"""

    x_screen: float
    wood: int = 0
    stone: int = 0
    structures: int = 0
    campfire: bool = False

    @property
    def built(self) -> bool:
        """社区是否已建成（至少一个结构）。"""
        return self.structures >= 1

    def need(self) -> str:
        """下一步缺什么："wood" / "stone" / "build"。"""
        if self.wood < 4:
            return "wood"
        if self.stone < 2:
            return "stone"
        return "build"

    def summary_zh(self) -> str:
        """社区进度摘要（中文，UI 展示用）。"""
        return (
            f"社区 木{self.wood}/4 石{self.stone}/2 结构{self.structures} "
            f"{'篝火' if self.campfire else ''}"
        )


class Technology:
    """原子网格上的采集、制作、配方与公共建造（世界规则层）。"""

    def __init__(self, config, scene, seed: int = 0) -> None:
        """初始化社区地点与统计。"""
        self.config = config
        self.scene = scene
        self.rng = random.Random(seed)
        self.community = CommunitySite(x_screen=config.width / 2.0)
        #: 本步验证通过、待文明层消费的设施 (kind, x_screen, agent_id)
        #: —— 见 Population
        self.pending_facilities: list[tuple[str, float, int]] = []
        #: 工具名 -> 已验证的自定义效果（有限组合式创新）
        self.custom_effects: dict[str, dict[str, float]] = {}
        self.recipes_executed = 0
        self.recipes_failed = 0

    # ------------------------------------------------------------------
    # 有限组合式创新：登记过的工具改变真实数值
    # ------------------------------------------------------------------
    def register_custom_effect(self, tool_name: str, effects: dict[str, float]) -> None:
        """登记组合工具**已验证**的效果。

        只有 :mod:`newworld.innovation` 在提案通过规则检查后才会调用；
        效果随后被 :meth:`harvest`、:meth:`simulate_recipe` 和
        :meth:`apply_facility_effects` 消费，让新技术改变真实世界
        数值而不是挂个标签。
        """
        name = str(tool_name or "").strip()
        if not name:
            raise ValueError("tool_name must be non-empty")
        clean: dict[str, float] = {}
        for key, value in dict(effects or {}).items():
            if key not in CUSTOM_EFFECT_KEYS:
                raise ValueError(f"unknown custom effect {key!r}")
            clean[str(key)] = float(value)
        self.custom_effects[name] = clean

    def custom_effect(self, tool_name: str | None, key: str) -> float:
        """查询某工具某项自定义效果的数值（无则 0.0）。"""
        if not tool_name:
            return 0.0
        return float(self.custom_effects.get(str(tool_name), {}).get(str(key), 0.0))

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    def _hand(self, character) -> tuple[float, float]:
        """手的世界坐标（身体前方 40px、脚上方 40px 处）。"""
        feet = character.y if character.y is not None else self.config.ground_y
        return character.x + character.facing * 40.0, feet - 40.0

    def nearest_resource(self, character, kind: str):
        """最近的树（木）、岩石（石）或植物（纤维），返回 (x, obj)。"""
        if kind == "fiber":
            # 绳纤维可以从任何植物采集，包括 agent 脚下的草：
            # 目标是最近的树/浆果丛，只剩地皮草时就是 agent 自己的位置
            best = None
            best_distance = 1e9
            for obj in self.scene.objects:
                if obj.kind not in ("tree", "bush"):
                    continue
                center = (obj.x0 + obj.x1) / 2.0 * materials.ATOM_SIZE
                distance = abs(center - character.x)
                if distance < best_distance:
                    best_distance = distance
                    best = (center, obj)
            return best if best is not None else (character.x, None)
        lookup = "wood" if kind == "wood" else kind
        best = None
        best_distance = 1e9
        for obj in self.scene.objects:
            if lookup == "wood" and obj.kind != "tree":
                continue
            if lookup == "stone" and obj.kind != "rock":
                continue
            center = (obj.x0 + obj.x1) / 2.0 * materials.ATOM_SIZE
            distance = abs(center - character.x)
            if distance < best_distance:
                best_distance = distance
                best = (center, obj)
        return best

    #: 每种可采集材料对应的材料原子（按顺序取第一个命中）
    _HARVEST_MATERIAL = {
        "wood": ("wood",),
        "stone": ("stone",),
        # 绳纤维来自树叶（树/浆果丛）或草
        "fiber": ("leaf", "grass"),
    }

    def harvest(self, character, kind: str) -> bool:
        """把手附近的材料原子摘进库存。

        ``fiber`` 采树叶原子（绳纤维），够不到叶时就采草；``wood``
        采树干原子、``stone`` 采岩石原子。够不到或库存已满时返回
        ``False``——调用方必须把它当作真实失败（P1 规划器就是如此）。
        """
        if kind not in self._HARVEST_MATERIAL:
            return False
        hand_x, hand_y = self._hand(character)
        cx = int(hand_x / materials.ATOM_SIZE)
        cy = int(hand_y / materials.ATOM_SIZE)
        targets = [materials.MATERIAL_INDEX[key] for key in self._HARVEST_MATERIAL[kind]]
        # 由内向外扩圈搜索，找最近的目标原子
        for radius in range(1, 10):
            best = None
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = cx + dx, cy + dy
                    if not (0 <= x < self.scene.aw and 0 <= y < self.scene.ah):
                        continue
                    if int(self.scene.grid[y, x]) in targets:
                        distance = dx * dx + dy * dy
                        if best is None or distance < best[0]:
                            best = (distance, x, y)
            if best is not None:
                _, x, y = best
                inventory = character.inventory
                # 有斧/镐时一次采 2 个；创新工具可再加成
                base = 2 if inventory.tool in ("axe", "pickaxe") else 1
                base += int(round(self.custom_effect(inventory.tool, "harvest_bonus")))
                amount = max(1, base)
                if not inventory.add(kind, amount):
                    return False  # 库存满：真实的失败
                self.scene.clear_atom(x, y)
                self.scene.refresh_caches()
                return True
        return False

    def near_community(self, character) -> bool:
        """是否站在旧版社区地点（90 px 内）。"""
        return abs(character.x - self.community.x_screen) < 90.0

    # ------------------------------------------------------------------
    # 配方（数据驱动；配方来自 knowledge.RECIPE_CATALOG）
    # ------------------------------------------------------------------
    def can_afford(self, character, recipe) -> bool:
        """库存是否备齐配方的全部投入。"""
        for kind, amount in dict(recipe.inputs).items():
            if not character.inventory.has(kind, amount):
                return False
        return True

    def has_tool_for(self, character, recipe) -> bool:
        """是否满足配方的工具前置（无前置恒为 True）。"""
        required = getattr(recipe, "requires_tool", None)
        if required is None:
            return True
        return character.inventory.tool == required or (
            required in ("any", "tool") and character.inventory.tool is not None
        )

    def recipe_blocker(self, character, recipe) -> str | None:
        """返回配方无法执行的简短原因码；能执行返回 None。

        公共设施可以在个体所站之处建造（建筑是一个*场所*：在半径
        内的人都受益），因此故意不设"必须到社区点"的前置——原来
        固定中央的地点只是兼容保留。
        """
        if not self.has_tool_for(character, recipe):
            return "missing_tool"
        for kind, amount in dict(recipe.inputs).items():
            if not character.inventory.has(kind, amount):
                return "missing_material"
        return None

    def execute_recipe(self, character, recipe) -> tuple[bool, str]:
        """模拟器权威的执行：扣投入、发产出。

        这是配方改变世界的*唯一*途径。规划器、教师、知识实验都走
        这里，因此神经网络或规划器都无法绕开规则。
        """
        blocker = self.recipe_blocker(character, recipe)
        if blocker is not None:
            self.recipes_failed += 1
            return False, blocker
        for kind, amount in dict(recipe.inputs).items():
            character.inventory.take(kind, amount)
        outputs = dict(recipe.outputs)
        for kind in (kind for kind in outputs if kind in FACILITY_OUTPUTS):
            # 设施材料被消耗；建筑本体排队，由文明层创建
            # （公共设施归文明层所有）
            self.pending_facilities.append((kind, character.x, character.agent_id))
        for kind, amount in outputs.items():
            if kind in INVENTORY_OUTPUTS:
                character.inventory.add(kind, amount)
            elif kind == "axe":
                character.inventory.tool = "axe"
            elif kind == "pickaxe":
                character.inventory.tool = "pickaxe"
            elif kind == "basket":
                character.inventory.basket += int(amount)
            elif kind in ("pot",):
                if character.inventory.tool is None:
                    character.inventory.tool = kind
        self.recipes_executed += 1
        return True, "ok"

    def simulate_recipe(self, character, recipe) -> tuple[bool, str]:
        """以**实验**方式尝试配方（KnowledgeStore 的验证路径）。

        与 :meth:`execute_recipe` 的区别：尝试可能*失败*——以
        ``recipe.difficulty`` 的概率材料报废（照样消耗），返回
        ``(False, "defect")``。这让知识发现成为一次有真实代价的
        实验，而不是保证成功的解锁。
        """
        blocker = self.recipe_blocker(character, recipe)
        if blocker is not None:
            self.recipes_failed += 1
            return False, blocker
        difficulty = float(getattr(recipe, "difficulty", 0.3))
        # 已验证的组合工具（例如绑好的斧柄）降低实验失败率
        success_bonus = self.custom_effect(character.inventory.tool, "craft_success_bonus")
        difficulty = max(0.0, difficulty - max(0.0, success_bonus))
        if self.rng.random() < difficulty:
            # 实验消耗了材料，却没得到可用的东西
            for kind, amount in dict(recipe.inputs).items():
                character.inventory.take(kind, amount)
            self.recipes_failed += 1
            return False, "defect"
        return self.execute_recipe(character, recipe)

    # ------------------------------------------------------------------
    # 旧版制作 / 社区（与原行为逐字节一致）
    # ------------------------------------------------------------------
    def craft(self, character) -> bool:
        """旧版配方：2 木 + 1 石 → 石斧。"""
        inventory = character.inventory
        if inventory.tool or inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        inventory.tool = "axe"
        return True

    def contribute(self, character) -> bool:
        """旧版捐献：2 木 + 1 石 → 社区进度。"""
        inventory = character.inventory
        if not self.near_community(character):
            return False
        if inventory.wood < 2 or inventory.stone < 1:
            return False
        inventory.wood -= 2
        inventory.stone -= 1
        community = self.community
        community.wood += 2
        community.stone += 1
        if community.wood >= 4 and community.stone >= 2 and community.structures < 1:
            community.structures = 1
            community.campfire = True
        return True

    def repair(self, character, facility) -> bool:
        """花材料恢复一个公共设施的状态。"""
        cost_wood = int(getattr(facility, "maintenance_wood", 0))
        cost_stone = int(getattr(facility, "maintenance_stone", 0))
        inventory = character.inventory
        if inventory.wood < cost_wood or inventory.stone < cost_stone:
            return False
        inventory.wood -= cost_wood
        inventory.stone -= cost_stone
        facility.maintained(character.agent_id, 0.0)
        return True

    # ------------------------------------------------------------------
    # 设施效果：技术必须让世界发生可见的改变
    # ------------------------------------------------------------------
    def apply_facility_effects(self, character, facilities, dt: float) -> dict[str, float]:
        """应用 ``character`` 附近所有**活跃**设施的效果。

        返回一个小字典说明发生了什么（telemetry/测试用）：
        ``{"warmth": 1, "rest": 1, "water": 1, "workshop": 1}``。

        多个设施的能量恢复有**上限**（``0.45/s``），堆同类建筑不能
        让个体永生；这个上限是世界规则，不是藏在规划器里的平衡旋钮。
        """
        effects: dict[str, float] = {}
        physiology = character.physiology
        energy_gain = 0.0
        # -- 已验证组合工具的被动效果（真实数值） ----------------------
        tool = character.inventory.tool
        tool_energy = self.custom_effect(tool, "energy_regen")
        tool_water = self.custom_effect(tool, "water_regen")
        if tool_energy > 0.0:
            physiology.energy = min(100.0, physiology.energy + tool_energy * dt)
            effects["tool_energy"] = effects.get("tool_energy", 0.0) + dt
        if tool_water > 0.0:
            physiology.hydration = min(100.0, physiology.hydration + tool_water * dt)
            effects["tool_water"] = effects.get("tool_water", 0.0) + dt
        for facility in facilities or ():
            if not getattr(facility, "active", True):
                continue
            radius = FACILITY_RADIUS.get(getattr(facility, "kind", ""), 0.0)
            if radius <= 0.0:
                continue
            if abs(getattr(facility, "x_screen", 0.0) - character.x) > radius:
                continue
            kind = facility.kind
            if kind == "campfire":
                # 篝火：取暖
                energy_gain += 0.35 * dt
                effects["warmth"] = effects.get("warmth", 0.0) + dt
            elif kind == "hut":
                # 小屋：睡觉/休息时恢复更快
                if physiology.sleeping or character.behavior_state == "rest":
                    energy_gain += 0.30 * dt
                    physiology.sleep_pressure = max(0.0, physiology.sleep_pressure - 0.30 * dt)
                effects["rest"] = effects.get("rest", 0.0) + dt
            elif kind == "well":
                # 水井：不用走到池塘就能喝水
                physiology.hydration = min(100.0, physiology.hydration + 1.2 * dt)
                effects["water"] = effects.get("water", 0.0) + dt
            elif kind == "workshop":
                # 工坊：本身无直接效果，只降低实验失败率（见 craft_failure_scale）
                effects["workshop"] = effects.get("workshop", 0.0) + dt
            elif kind == "granary":
                # 粮仓：存粮让附近所有人缓慢被动回能
                energy_gain += 0.22 * dt
                effects["granary"] = effects.get("granary", 0.0) + dt
        if energy_gain > 0.0:
            # 能量恢复封顶 0.45/s：防止堆设施不死
            physiology.energy = min(100.0, physiology.energy + min(energy_gain, 0.45 * dt))
        return effects

    def craft_failure_scale(self, character, facilities) -> float:
        """工坊把实验失败概率减半；返回缩放系数（1.0 = 无工坊）。"""
        for facility in facilities or ():
            if getattr(facility, "kind", "") != "workshop":
                continue
            if not getattr(facility, "active", True):
                continue
            if abs(facility.x_screen - character.x) <= FACILITY_RADIUS["workshop"]:
                return 0.5
        return 1.0

    # ------------------------------------------------------------------
    def summary_zh(self) -> str:
        """技术层摘要（中文，实为社区进度）。"""
        return self.community.summary_zh()
