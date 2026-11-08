# 已确认12题政策版本

基于冻结r7-product，落实用户最后答复：两道刀刃题均要求未开刃，其余4组采用此前建议。六个顶层维度保持不变。

新增price_basis=sale_unit表示一个经商品证据核验的销售单位，不编造物理分母；新增composition/member_groups/matches_members及member_ranges表达一套商品各成员的范围。仅这批已确认的整十“多段”组合采用21～29等范围，不把一般近似尺寸统一套用此政策。

旧三四片、并列至少、成员范围均有来源检查和反例；无task_id检查绕过。逐题数据补丁与通用检查代码分开，1000份原query未变。当前1000/1000程序通过，整批语义批准仍未放行。

详见[落实报告](../../analysis/semantic_final12_20261107/12题政策落实报告.md)，最新路径见[状态索引](../../analysis/semantic_latest.json)。
