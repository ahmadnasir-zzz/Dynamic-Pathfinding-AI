Dynamic AI Pathfinding Agent

A real-time, interactive pathfinding visualizer built with Python and Tkinter. 

This project demonstrates informed search algorithms in both static and unpredictable, dynamic environments.
🌟 Key Features
Algorithm Comparison: Toggle between A Search* ($f = g + h$) and Greedy BFS ($f = h$).
Dynamic Re-planning: Real-time obstacle spawning forces the agent to recalculate paths mid-movement.
Custom Heuristics: Compare Manhattan and Euclidean distance patterns.
Interactive Grid: Drag-to-draw walls, randomize mazes, and reposition Start/Goal nodes.
Live Metrics: Track Nodes Expanded, Path Cost, and Execution Time.

🛠️ Technical StackLanguage: Python 3.xGUI: Tkinter (Standard Library)

Data Structures: Priority Queues (heapq) for $O(\log N)$ frontier management.

⚙️ Installation & ExecutionThis project uses only Standard Python Libraries.

No external dependencies are required.

🎮 How to UseDesign: Use Draw Walls or Random Maze to build the environment.

Configure: Select an Algorithm and Heuristic from the sidebar.
Simulate: 
Run Search: Visualizes the expansion of the search frontier.
Dynamic Mode: Watch the agent navigate while the environment changes.
Analyze: Compare performance via the Metrics panel.
