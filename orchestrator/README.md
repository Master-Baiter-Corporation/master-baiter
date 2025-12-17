# Useful commands

Publish std_msgs:
```sh
ros2 topic pub --once /baiter_cmd std_msgs/String 'data: avance'
```

Publish geometry_msgs:
```sh
ros2 topic pub --once /baiter_goal_pos geometry_msgs/Point "{x: 1.0, y: 2.4, z: 0.0}"
```
