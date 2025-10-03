import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 使用交互式查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 保持窗口打开一段时间，或模拟一段时间
    time.sleep(10)
