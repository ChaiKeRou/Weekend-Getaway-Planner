# Weekend-Getaway-Planner

## Project Overview
The Weekend Getaway Planner is designed to help students organize carpooling for weekend trips. It uses a flow network algorithm to allocate students to cars or destinations based on their preferences and available drivers. The project involves creating vertices and edges in a flow network, transforming the network to remove demands, and applying the Ford-Fulkerson algorithm to find the maximum flow.

## Features
- Vertex Class: Represents a vertex in the flow network with attributes such as role, demand, and driver status.
- Edge Class: Represents an edge in the flow network with methods to update flow and set forward/backward edges.
- Flow Network Class: Manages the entire flow network, including initializing vertices and edges, removing demands, and building the residual network.

## How to Run the Code
1. Prerequisites: Ensure you have Python installed on your system.

2. Files:

3. main.py: Contains the main implementation of the project.
README.txt: This file containing an overview and instructions.
Running the Code:

Open a terminal or command prompt.
Navigate to the directory containing main.py.
Run the following command:
python main.py

## Code Explanation
The project is structured into three main classes:

Vertex Class
The Vertex class represents a vertex in the flow network. It has methods to add edges, set roles, set demands, check if a vertex is a driver, visit and discover vertices, and reset for BFS.

Edge Class
The Edge class represents an edge in the flow network. It has methods to initialize forward and backward edges (for the residual network) and update the flow of the edge.

Flow Network Class
The FlowNetwork class manages the flow network. It initializes the network, removes demands to transform it into a flow network, builds the residual network, and applies the Ford-Fulkerson algorithm for finding the maximum flow.

## Credits
Author: Chai Ke Rou
Date: October 2023