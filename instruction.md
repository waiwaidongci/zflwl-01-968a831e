# 分支馈线网络档案复原

城北一条低压馈线刚做完支线改造，台区图却在迁移系统时丢了。十二块客户表的逐时段读数、
六台柱上变压器的窗口电量还留在采集库，馈线首端三相功率表也能导出；但客户表与变压器的
对应关系、每台变压器落在哪一相、相内各段电缆怎样接到上游节点，都没有记录。线路单位长度
损耗系数仍按三相统一整定，现场只知道它没有改过。

候选程序位于 `/app/predict.py`，目前只交一张空网络。入口仍按 `<input.json> <output.json>`
接收两个路径，计算只用镜像内的 Python 标准库。

## 采集记录

输入外层对象有且只有 `schema`、`periods`、`tolerance_kw`、`energy_tolerance_kwh`、
`meters`、`transformers`、`spans`、`channels` 八个字段。
`schema` 固定写成 `"feeder-tree-v3"`。`periods` 是 1 到 6 的整数。
`tolerance_kw` 是 0 到 0.005 闭区间内的实数；`energy_tolerance_kwh` 是 0 到 0.01
闭区间内的实数。布尔值不能作为数值。

`meters` 有 6 到 12 个表对象，每个对象有且只有 `id` 和 `profile`。
表号写成 `M001` 这种三位数字形式，同一输入内不得重号。`profile` 有 `periods` 个非负
有限数值，单位为 kW，表示一块表在对应 15 分钟时段内的平均有功功率；布尔值不能作为
功率。第 `t` 个时段的一块表在采集窗口内的电量按下式换算，15 分钟折合 0.25 小时：

```text
energy_t = 0.25 * profile_t
```

`transformers` 有 3 到 6 台设备，每台有且只有 `id`、`service_points`、`energy_kwh`、
`copper_loss_kw_inv` 四个字段。编号写成 `T01` 这种两位数字形式，同一输入内不得重号。
`service_points` 是 1 到 2 的整数，全部服务点之和等于表数。
`energy_kwh` 是采集窗口内的变压器供电量，非负有限；`copper_loss_kw_inv` 是
0.0000001 到 0.01 的有限实数。

`spans` 给出所有可敷设的定向电缆段，每段有且只有 `parent`、`child`、`length_m`。
`parent` 是 `A`、`B`、`C` 或一台变压器编号，`child` 只能是变压器编号。
同一父节点到同一子节点不得重复，长度是 0.1 到 1000 的有限实数。
三个相根到每台变压器、以及任意两台变压器之间的定向电缆段都要在 `spans` 中出现。

`channels` 有且只有 `A`、`B`、`C` 三个字段。每个数组有 `periods` 个非负有限数值，
表示馈线首端对应相在一个时段内的有功功率，数组长度都等于 `periods`。

## 网络关系

一块表只接一台变压器，变压器承接的表数等于它的 `service_points`。一台变压器承接的
全部表的窗口电量之和，按 `energy_tolerance_kwh` 与该变压器的 `energy_kwh` 对齐；其中
每块表的窗口电量按上面的 0.25 小时换算。
每台变压器只接一相，三相各自至少有一台变压器。
每台变压器恰好选择一条来自相根或另一台变压器的上游电缆；沿父节点继续向上必须回到同一
相根，不能形成环。不同变压器可以接到同一个上游节点，因此一相可以是分支树，而不是单链。

记变压器 `S` 的负荷为 `load_S[t]`，铭牌系数为 `c_S`，线路单位长度系数为 `g`：

```text
copper_S[t] = c_S * load_S[t]^2
inlet_S[t]  = load_S[t] + copper_S[t]
```

对一条父节点为 `P`、子节点为 `S` 的电缆，`carry_S[t]` 是 `S` 及其全部下游变压器的
`inlet` 之和，该段损耗为 `length(P,S) * g * carry_S[t]^2`。
相根 `A`、`B`、`C` 分别视为三相线路的首端。每一相的首端读数等于本相全部
`inlet_S[t]` 加本相全部电缆段损耗；每一相、每一时段的预测值与 `channels` 之差不得超过
`tolerance_kw`。

`g` 是 0.0000001 到 0.001 的实数，三相共用同一个值。
某相在时段 `t` 的损耗等于本相变压器铜损与本相全部电缆段损耗之和。

## 结果

若客户表归属、变压器相别、定向父节点和共用线路系数只有一组满足全部关系，输出对象有且
只有 `status`、`meter_to_transformer`、`transformer_phase`、`connections`、
`losses_kw`、`line_loss_kw_inv` 六个字段。

`status` 为 `"ok"`。`meter_to_transformer` 的键是全部表号，值是变压器编号。
`transformer_phase` 的键是全部变压器编号，值是 `A`、`B`、`C` 之一。
`connections` 是全部变压器对应的父节点、子节点对象，按 `child` 字典序排列，每个对象
有且只有 `parent` 和 `child` 两个字段。`losses_kw` 有且只有 `A`、`B`、`C` 三个字段，
每个值是按 `periods` 给出的相损耗数组，数组元素保留六位小数。
`line_loss_kw_inv` 保留八位小数。

若存在两组或更多不同网络，输出对象有且只有 `status` 和 `candidates` 两个字段。
`status` 为 `"ambiguous"`，`candidates` 固定为整数 2，不在第二组之后继续列举。

输入结构、类型或值域不合法，或者没有任何网络能满足读数时，本次复盘失败，不写输出文件；
标准错误输出一行以 `REFUSED: ` 开头的原因，退出码为 2。

输出 JSON 使用紧凑格式，键按字典序排列，末尾带一个换行。程序不读取测试目录，源码里
不写死某组表号或网络答案。同一份改造记录会被反复复盘；重跑必须得到相同的结果字节。
