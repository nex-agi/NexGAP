任务说明 — 将需要合成query的框架在 Gyrfalcon 框架中配置好
目标

1. Gyrfalcon 框架用于为其它框架合成 query，因此需理解 Gyrfalcon 的工作流与配置要求，保证被集成框架能被 Gyrfalcon 正确识别和驱动。
2. 阅读并理解源框架：/path/to/NexGAP/NexA4A/src/created_frameworks/sync_sleuth_music_rights
3. 将该框架在 Gyrfalcon 框架中配置好：/path/to/NexGAP/Gyrfalcon，你也需要去阅读这个框架，理解他的逻辑workflow
   - 配置后，框架应出现在：/path/to/NexGAP/Gyrfalcon/frameworks/下的一个以该框架名命名的文件夹中

交付物（必须包含且直接放在目标路径下）

1. 完整的框架目录：/path/to/NexGAP/Gyrfalcon/frameworks/下的一个以该框架名命名的文件夹中，注意框架名需要与原来一致
2. config.json文件（参考Gyrfalcon 中其他框架）
3. framework config 文件（与 Gyrfalcon 中其他框架的字段一模一样）
   - 请严格对齐现有框架的字段名、数据类型和必需项
   - description 字段必须写得详尽，清晰说明该 agent 能完成的任务、适用场景, 但是注意不要再description中出现他有哪些agent以及有哪些tool
4. persona.jsonl
   - 必须包含 500 条不同 persona，每行为一个 JSON 对象
   - persona 内容应丰富、多样，覆盖所有可能使用该 agent 的角色/行业
   - 中英文应该区分开，不要出现中英混合的内容，描述要正常，符合人类说法
5. problem_types
   - 包含这个框架能解决的各种问题类型，可以参考其他框架的例子
