from __future__ import annotations
from collections import deque 
import math

#Weekend Getaway Planner
class Vertex:
    """
    A vertex that exists in the flow network.
    - add_edge: Adds an edge to the vertex.
    - set_role: Sets the role of the vertex.
    - set_demand: Sets the demand of the vertex.
    - set_driver: Sets the vertex as a driver.
    - is_driver: Returns whether the vertex is a driver.
    - visit: Visits the vertex.
    - discover: Discovers the vertex.
    - bfs_reset: Resets the vertex to run bfs multiple times.

    Attributes:
        - id: The id of the vertex, which is the index in the graph adjacency list. (int)
        - role: The role of the vertex, which is the student number or car/destination number. (int)
        - edges: A list of edges that are connected from the vertex to the other vertices. ([Edge])
        - driver: A boolean value indicating whether the vertex is a driver. (bool)
        - demand: The demand of the vertex. (int)
        - visited: A boolean value indicating whether the vertex has been visited. (bool)
        - discovered: A boolean value indicating whether the vertex has been discovered. (bool)
        - parent: The parent of the vertex for bfs backtracking. (Vertex)
    """
    def __init__(self, id: int) -> None:
        """
        Function description:
        Initialises a vertex in the flow network.

        :Input:
        id: The id of the vertex, which is the index in the graph adjacency list. (int)

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.id = id
        self.role = None
        self.edges = []
        self.driver = False
        self.demand = 0
        self.visited = False
        self.discovered = False
        self.parent = None
    
    def add_edge(self, edge: Edge) -> None:
        """
        Function description:
        Adds an edge to the vertex.
        
        :Input:
        edge: The edge connected from the vertex to other vertices. (Edge)
        
        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.edges.append(edge)
    
    def set_role(self, role: int) -> None:
        """
        Function description:
        Sets the role of the vertex.

        :Input:
        role: The role of the vertex, which is the student number or car/destination number. (int)

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.role = role

    def set_demand(self, demand: int) -> None:
        """
        Function description:
        Sets the demand of the vertex.

        :Input:
        demand: The demand of the vertex. (int)

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.demand = demand

    def set_driver(self) -> None:
        """
        Function description:
        Sets the vertex as a driver.

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.driver = True

    def is_driver(self) -> bool:
        """
        Function description:
        Returns whether the vertex is a driver.

        :Output, return or postcondition:
        A boolean value indicating whether the vertex is a driver. (bool)

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        return self.driver

    def visit(self) -> None:
        """
        Function description:
        Visits the vertex.

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.visited = True
    
    def discover(self) -> None:
        """
        Function description:
        Discovers the vertex.

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.discovered = True

    def bfs_reset(self) -> None:
        """
        Function description:
        Resets the vertex to run bfs multiple times.

        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.visited = False
        self.discovered = False
        self.parent = None

class Edge:
    """
    An edge that exists in the flow network.
    - set_forward_backward: Initialises the forward and backward edges(residual network) of the edge(flow network).
    - update_flow: Updates the flow of the edge in flow network and its forward & backward edgse in residual network.

    Attributes:
    - u: The vertex that the edge is connected from. (Vertex)
    - v: The vertex that the edge is connected to. (Vertex)
    - capacity: The capacity of the edge. (int)
    - flow: The flow of the edge. (int)
    - direction: A boolean value indicating whether the edge is forward or backward. (bool)
        - True: The edge is forward.
        - False: The edge is backward.
    - forward: The forward edge of this edge in residual network. (Edge)
    - backward: The backward edge of this edge in residual network. (Edge)
    """
    def __init__(self, u: Vertex, v: Vertex, capacity: int, flow: int=0, direction: bool=True) -> None:
        """
        Function description:
        Initialises an edge in the flow network.

        :Input:
        u: The vertex that the edge is connected from. (Vertex)
        v: The vertex that the edge is connected to. (Vertex)
        capacity: The capacity of the edge. (int)
        flow: The flow of the edge. (int)
        direction: A boolean value indicating whether the edge is forward or backward. (bool)
        
        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = flow
        self.direction = direction

    def set_forward_backward(self, u: Vertex, v: Vertex) -> (Edge, Edge):
        """
        Function description:
        Initialises the forward and backward edges(residual network) of the edge(flow network).
        Forward edge: 
            - capacity: same as the edge (flow network)
            - flow: the capacity minus the flow of the edge (flow network)
            - direction: True
        Backward edge:
            - capacity: same as the edge (flow network)
            - flow: same as the flow of the edge (flow network)
            - direction: False

        :Input:
        u: The vertex that the edge is connected from in flow network. (Vertex)
        v: The vertex that the edge is connected to in flow network. (Vertex)

        :Output, return or postcondition:
        A tuple of the forward and backward edges. ((Edge, Edge))
    
        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        self.forward = Edge(u, v, self.capacity, self.capacity-self.flow)
        self.backward = Edge(v, u, self.capacity, self.flow, False)
        return (self.forward, self.backward)

    def update_flow(self, flow: int, direction: int) -> None:
        """
        Function description:
        Updates the flow of the edge in flow network and its forward & backward edgse in residual network.

        :Input:
        flow: The flow to be updated. (int)
        direction: A boolean value indicating whether the forward/backward of the edge 
                    is used to traverse from source to target. (bool)
            - True: Forward edge is used in the bfs traversal.
            - False: Backward edge is used in the bfs traversal.
        
        :Complexity:
        :Time complexity:
            - Best-Case = Worst-Case : O(1)
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)
        """
        #if forward edge is used in bfs
        if direction == True:
            # add flow to the edge, but not exceed the capacity
            f = self.flow + flow
            if f > self.capacity:
                f = self.capacity
            self.flow = f
        #else if backward edge is used in bfs
        elif direction == False:
            # subtract flow from the edge, will never below 0
            self.flow -= flow
        # update the flow of the forward and backward edges in residual network
        self.forward.flow = self.capacity-self.flow
        self.backward.flow = self.flow
        
class FlowNetwork:
    """
    A flow network.
    - remove_demand: Removes the demand of the vertices.
    - build_residual_network: Builds the residual network.
    - bfs: Performs bfs on the flow network.
    - bfs_reset: Resets the flow network to run bfs multiple times.
    - ford_fulkerson: Performs Ford-Fulkerson algorithm on the flow network.

    Attributes:
        - n: The number of students. (int)
        - num_c_d: The number of cars/destinations. (int)
        - num_vertex: The number of vertices in the flow network. (int)
        - graph: A list of vertices in the flow network. ([Vertex])
        - residual_network: A list of vertices in the residual network. ([Vertex])
    """
    def __init__(self, preferences: [[int]], drivers: [int]) -> None:
        """
        Function description:
        Initialises a flow network staring from circulation with demands. 
        
        Approach description:
        - Initialises all required vertices.
            - n vertices for n students
                - In the adjacency list of the graph, [0...n-1] for the students
            - (n//5) vertices for the students with licence of each car(s)/desniation(s) [extra] 
                - In the adjacency list of the graph, [n...n+(n//5)-1] for the students with licence
            - (n//5) vertices for the students without licence of each car(s)/desniation(s) [extra]
                - In the adjacency list of the graph, [n+(n//5)...n+2*(n//5)-1] for the students without licence
            - (n//5) vertices for (n//5) cars/destinations
                - In the adjacency list of the graph, [n+2*(n//5)...n+3*(n//5)-1] for the cars/destinations
            - 1 for the sum of the total number of students that has been allocated [extra]
                - In the adjacency list of the graph, [-3] for the sum
            - 1 for the source and 1 for the target
                - In the adjacency list of the graph, [-2] for the source, [-1] for the target
        - Sets the demand of the sum vertex to total number of students minus the required number of drivers
            (2 per car/destination), which means it has a demand of n-(2*(n//5)) as every students must be allocated.
        - Marks the students with licence to be drivers.
        - Sets the role & demand of the students and connects n students to thier prefered car(s)/desniation(s) 
            - The student role is the student number.
            - The student demand is -1 which means it has a supply of 1 as every student can only be allocated to 1 car/destination.
            - Connects the students according to its preferences and with licence or without licence to the 
                respective vertices with a capacity of 1.
            - Set the demand of the vertices for the students with licence of each car(s)/desniation(s) to 2,
                which means that each car(s)/desniation(s) has a demand of 2 drivers.
        - For every extra vertices of cars/destinations, 
            - Set the role of the vertices to be the car/destination number.
            - Connects the vertices for the students with licence of each car(s)/desniation(s) [extra] 
                and without licence to the car/destination vertices with a capacity of 3.
            - Connects the vertices for (n//5) cars/destinations to the sum vertex with a capacity of 3.
        
        :Input:
        preferences: A list of lists containing the preferences of the students. ([[int]])
        drivers: A list containing the car/destination numbers. ([int])

        :Complexity: Assume N is the number of students, D is the number of drivers, 
                        C is the number of cars/destinations.
        :Time complexity:
            - Initialisation of a list of vertices in the flow network & residual network: O(3 + N + 3C) = O(N+C)
            - Marking the students with licence to be drivers: O(D) 
            - Setting the role & demand of the students and connecting them to thier prefered car(s)/desniation(s): 
                - Best-Case: O(N), when each student has only 1 preference
                - Worst-Case: O(NC), when each student prefers to go for every cars/destinations
            - Connecting the extra vertices the respective cars/destinations and the cars/destinations to the sum vertex: O(C)
            - Best-Case = O(N+C+D+N+C) = O(N) as D <= N, C <= N//5
            - Worst-Case = O(N+C+D+NC+C) = O(N^2) as D <= N, C <= N//5
        :Aux space complexity:
            - Initialisation of a list of vertices in the flow network & residual network: O(2N)
            - Initialisation of a list of edges for all vertices: O(NC+3C)
            - Best-Case = Worst-Case : O(2N+NC+3C) = O(N^2) as D <= N, C <= N//5
        :Total space complexity:
            - Input space complexity: O(NC+D)
                - An input list of preferences, each student can have at most C preferences: O(NC)
                - An input list of drivers: O(D)
            - Aux space complexity: O(N^2)
            - Total: O(NC+D+N^2) = O(N^2) as D <= N, C <= N//5
        """
        self.n = len(preferences)
        self.num_c_d = math.ceil(self.n/5)              #number of cars/destinations
        self.num_vertex = 3 + self.n + (3*self.num_c_d) #n for students, 2*(n//5) for extra nodes, (n//5) for cars/destinations, 1 for sum, 2 for source and sink 
        
        # construct the graph, initialise the vertices
        self.graph = [None] * self.num_vertex 
        self.residual_network = [None] * self.num_vertex
        for i in range(self.num_vertex):
            self.graph[i] = Vertex(i)
            self.residual_network[i] = Vertex(i)

        # set the demand of the sum node to total number of students minus the required number of drivers(2 per car/destination)
        self.graph[-3].set_demand(self.n-(2*self.num_c_d))

        # mark the students with licence to be drivers
        for d in drivers:
            self.graph[d].set_driver()

        # set the role & demand of the students and connect them to thier prefered car(s)/desniation(s)
        for i in range(self.n): 
            stud = self.graph[i]
            stud.set_role(i)
            stud.set_demand(-1)
            preference = preferences[i]
            if stud.is_driver():
                for p in preference:
                    self.graph[p+self.n].set_demand(2)
                    stud.add_edge(Edge(stud,self.graph[p+self.n],1))
            else:
                for p in preference:
                    stud.add_edge(Edge(stud,self.graph[p+self.n+self.num_c_d],1))

        # connect the extra vertices the respective cars/destinations and the cars/destinations to the sum vertex
        for i in range(self.n, self.n+self.num_c_d):
            extra_driver = self.graph[i]
            extra = self.graph[i+self.num_c_d]
            c_d = self.graph[i+2*self.num_c_d]
            #set the role of the extra vertices to be the car/destination number
            extra_driver.set_role(i-self.n)
            extra.set_role(i-self.n)
            #create edges between the extra vertices and the car/destination vertices
            extra_driver.add_edge(Edge(extra_driver,c_d,3))
            extra.add_edge(Edge(extra,c_d,3))
            c_d.add_edge(Edge(c_d, self.graph[-3], 3))

    def remove_demand(self) -> None:
        """
        Function description:
        Removes the demand of the vertices in circulation with demands to transfrom it to a flow network by connecting 
        the vertices with negative demand to the source and the vertices with positive demand to the target. 

        :Complexity: Assume N is the number of students, C is the number of cars/destinations.
        :Time complexity:
            - Loop through all vertices in the graph to remove the vertices with demand: O(N)
            - Best-Case = Worst-Case : O(N)
        :Aux space complexity:
            - Intialise the edges that connect the vertices with negative demand (N students) to the source: O(N)
            - Intialise the edges that connect the vertices with positive demand (C cars/destinations with driver[extra]) 
                 and to the target: O(C)
            - Initialise the edge that connect the sum vertex either to the source or the target: O(1)
            - Best-Case = Worst-Case : O(N+C+1) = O(N), as C <= N//5
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(N)
            - Total: O(1+N) = O(N)
        """
        source = self.graph[-2]
        target = self.graph[-1]
        # loop through all vertices in the graph to remove the vertices with demand
        for i in range(self.num_vertex):
            vertex = self.graph[i]
            # if the vertex has negative demand, connect it to the source with a capacity of its demand
            if vertex.demand < 0: 
                source.add_edge(Edge(source,vertex,-vertex.demand))
            # if the vertex has positive demand, connect it to the target with a capacity of its demand
            elif vertex.demand > 0:
                vertex.add_edge(Edge(vertex,target,vertex.demand))
            vertex.set_demand(0) #remove the demand of the vertex

    def build_residual_network(self) -> None:
        """
        Function description:
        Builds the residual network for the flow netword by adding the forward and backward edges to the 
        residual network.

        :Complexity: Assume N is the number of students, C is the number of cars/destinations.
        :Time complexity:
            - Loop through all edges in the flow network to build the residual network: O(NC+3C+N+C+1) = O(NC+4C+N+1)
            - Best-Case = Worst-Case : O(NC+4C+N+1) = O(N^2), as C <= N//5
        :Aux space complexity:
            - Intialise the forward and backward edges for all edges in the flow network: O(2*(NC+3C+N+C+1))
            - Best-Case = Worst-Case : O(2*(NC+3C+N+C+1)) = O(N^2), as C <= N//5
        : Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(N^2)
            - Total: O(1+N^2) = O(N^2)
        """
        for i in range(self.num_vertex):
            ori_vertex = self.graph[i]
            vertex = self.residual_network[i]
            for edge in ori_vertex.edges:
                #get the same vertex in the residual network
                u = self.residual_network[edge.u.id]    
                v = self.residual_network[edge.v.id]
                #set the forward and backward edges in the residual network    
                forward_edge, backward_edge = edge.set_forward_backward(u, v)
                u.add_edge(forward_edge)
                v.add_edge(backward_edge)

    def bfs(self, source: Vertex, target: Vertex) -> bool:
        """
        Function description:
        Performs bfs on the residual network to determine whether there is an augmenting path from source to target.

        Approach description:
        - Initialise a deque which have the same complexity of popping the first element as queue to store the vertices to be discovered.
        - Use a while loop to loop through all vertices in the queue until a path from source to target is found.
        - Set the current vertex as the parent of its next vertex if it is discovered for backtracking purpose.
        - If the target vertex is discovered, means there is a path from source to target, return True.
        - After discovering all vertices starting from source still can't discover the target, return False.

        :Input:
        source: The source vertex in the residual network. (Vertex)
        target: The target vertex in the residual network. (Vertex)

        :Output, return or postcondition:
        A boolean value indicating whether there is an augmenting path from source to target in the residual network. (bool)
            - True: There is a path from source to target.
            - False: There is no path from source to target.
        
        :Complexity: Assume N is the number of students, C is the number of cars/destinations.
        :Time complexity:
            - Loop through all vertices in the residual network to perform bfs: O(NC+4C+N+1)
            - Best-Case = Worst-Case : O(NC+4C+N+1) = O(N^2), as C <= N//5
        :Aux space complexity:
            - Create a queue to store the vertices to be visited: O(NC+4C+N+1)

        """
        discovered = deque([])         #use deque for discovered queue
        discovered.append(source)
        while len(discovered) > 0:
            #means visited, removing from discovered
            u = discovered.popleft()   #O(1), pop the first element in deque is O(1)
            u.visit()
            for edge in u.edges:
                if edge.flow != 0:              
                    v = edge.v
                    if v == target:
                        v.parent = u   #set the parent of the target vertex to the current vertex
                        return True
                    elif v.discovered == False: 
                        #means discovered, adding to discovered
                        v.parent = u   #set the parent of the vertex to the current vertex
                        discovered.append(v)
                        v.discover()
        return False

    def bfs_reset(self) -> None:
        """
        Function description:
        Resets the residual network to run bfs multiple times.
        
        :Complexity: Assume N is the number of students.
        :Time complexity:
            - Loop through all vertices in the residual network to reset the vertices: O(3 + N + 3C)
            - Best-Case = Worst-Case : O(3 + N + 3C) = O(N), as C <= N//5
        :Aux space complexity:
            - Best-Case = Worst-Case : O(1)
        :Total space complexity:
            - Input space complexity: O(1)
            - Aux space complexity: O(1)
            - Total: O(1+1) = O(1)    
        """
        for i in range(self.num_vertex):
            self.residual_network[i].bfs_reset()

    def ford_fulkerson(self) -> bool:
        """
        Function description:
        Performs Ford-Fulkerson algorithm on the flow network to find a way that can allocate all students 
        to the cars/destinations.

        """
        self.remove_demand()            #remove the demand of the vertices, circulation with demands -> flow network
        self.build_residual_network()   #build the residual network for finding augmenting path
        
        max_flow = 0
        source = self.residual_network[-2]
        target = self.residual_network[-1]

        #loop until no augmenting path can be found in the residual network
        while self.bfs(source, target) == True:
            cur = target
            cur_parent = target.parent
            largest_bn = float("inf")   
            direction = None
            path = [] #for backtracking, store (v, direction) of the path that is used in bfs traversal

            #backtracking to find the largest bottleneck (edmunds-karp)
            while cur != source:
                for edge in cur_parent.edges:
                    if edge.v == cur:
                        if edge.flow < largest_bn:
                            largest_bn = edge.flow
                        backtrack_edge = edge
                        # direction = edge.direction 
                path.append(backtrack_edge)
                # path.append((cur, direction))
                cur = cur_parent                    
                cur_parent = cur_parent.parent
            
            #update the max flow, by adding the flow that has been flowed through the path
            max_flow += largest_bn

            # #update the flow network
            # start = self.graph[source.id]
            # for i in range(len(path)-1,-1,-1):       
            #     next_vertex = self.graph[path[i][0].id]
            #     #if the edge used for bfs is the forward edge in residual network, update the flow of the edge in flow network
            #     for edge in start.edges:
            #         if edge.v == next_vertex:
            #             edge.update_flow(largest_bn, path[i][1])
            #             break
            #     #if the edge used for bfs is the backward edge in residual network, update the flow of the edge in flow network
            #     for edge in next_vertex.edges:
            #         if edge.v == start:
            #             edge.update_flow(largest_bn, path[i][1])
            #             break
            #     start = next_vertex
            for p in path:
                if p.direction == True:
                    u = self.graph[p.u.id]
                    v = self.graph[p.v.id]
                elif p.direction == False:
                    u = self.graph[p.v.id]
                    v = self.graph[p.u.id]
                for edge in u.edges:
                    if edge.v == v:
                        edge.update_flow(largest_bn, p.direction)
                        break
            self.bfs_reset()   #reset the residual network to run bfs multiple times

        #if there is less than 2 students, cannot allocate (at least 2 students with licences per car), return False
        if max_flow < 2:        
            return False
        #else if all students can be allocated, return True
        elif max_flow == self.n:
            return True
        #else not all students can be allocated, return False
        else:
            return False

def allocate(preferences: [[int]], licences: [int]) -> [[int]]:
    """
    Function description:
    Allocates the students to the cars/destinations based on their preferences and licences.

    :Input:
    preferences: A list of lists containing the preferences of the students. ([[int]])
    licences: A list containing the car/destination numbers of the students with licences. ([int])

    :Output, return or postcondition:
    A list of lists containing the students allocated to the cars/destinations. ([[int],...])
        - The index of the list is the car/destination number.
        - The elements in the list are the student numbers allocated to the car/destination.
    
    :Complexity: Assume N is the number of students, C is the number of cars/destinations.
    :Time complexity:
        - Construct the graph: 
        - Ford-Fulkerson algorithm:
        -
        - Best-Case = Worst-Case : O(NM)
    :Aux space complexity:
        - Construct the graph: O(N+
    :Total space complexity:
        - Input space complexity: O()
        - Aux space complexity: O(N+M)
        - Total: O(N+M+N+M+N+M) = O(3N+3M) = O(N+M)
    """
    #construct a graph based on the requirements
    g = FlowNetwork(preferences, licences)
    #a list of list to store the allocation result for each car/destination
    ret = []    
    for i in range(g.num_c_d):
        ret.append([])
    #if all students can be allocated, return the allocation result
    if g.ford_fulkerson():
        #loop through all students to find the car/destination they are allocated to
        for i in range(g.n):
            stud = g.graph[i]
            #loop through all edges of the student to find the edge with flow of 1 which means 
            # the student is allocated to the car/destination that the edge is connected to
            for edge in stud.edges:
                if edge.flow == 1:
                    j = edge.v.role         #obtain the car/destination number
                    ret[j].append(stud.id)  #add the student number to the list of the car/destination
                    break                   #break the loop as the student is allocated, since each student can only be allocated to one car/destination
        return ret
    #else cannot allocate all students by fulfilling all conditions, return None
    else:
        return None
                

if __name__ == "__main__":
    preferences = [[0], [1], [0,1], [0, 1], [1, 0], [1], [1, 0], [0, 1], [1], [1]]
    licences = [1, 4, 0, 5, 8]
    print(allocate(preferences, licences))
    preferences = [[1], [0], [0, 1], [1, 0], [0, 1], [1, 0], [1], [1], [0], [0, 1]]
    licences = [6, 3, 4, 9, 1]
    print(allocate(preferences, licences))
    preferences = [[0],[0]]
    licences = [0,1]
    print(allocate(preferences, licences))