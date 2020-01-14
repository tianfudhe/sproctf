# -*- coding: UTF-8 -*-
# Description: Loading and simplify network from raw road network files
# Author: Xianyuan Zhan

import networkx as nx
import math
import json
from vincenty import vincenty


class GD_Node:
    def __init__(self, nodeLine):
        self.id = int(nodeLine.split('\t')[0])
        self.lat = float(nodeLine.split('\t')[1])
        self.lng = float(nodeLine.split('\t')[2])
        self.str = nodeLine

    def updateID(self, new_id):
        self.id = new_id
        self.str = str(new_id)+'\t'+self.str.split('\t', 1)[1]

    def updateStr(self):
        self.str = "{}\t{}\t{}".format(self.id, self.lat, self.lng)


class GD_Edge:
    def __init__(self, infoLine, coordLine):
        eles = infoLine.split('\t')

        self.id = int(eles[0])
        self.infoLine = infoLine
        self.coordLine = coordLine

        self.startNode = int(eles[1])
        self.endNode = int(eles[2])
        self.length = float(eles[3])
        self.dir = int(eles[4])
        self.maxLane = int(eles[5])
        self.maxSpeed = int(eles[6])
        self.level = int(eles[7])
        self.name = eles[8]

    def appendNewStart(self, n):
        self.startNode = n.id
        self.length += vincenty(self.getStartCoord(), [n.lat, n.lng])
        self.coordLine = '{} {},'.format(n.lat, n.lng)+self.coordLine
        self.generateInfoLine()

    def appendNewEnd(self, n):
        self.endNode = n.id
        self.length += vincenty(self.getEndCoord(), [n.lat, n.lng])
        self.coordLine = self.coordLine+',{} {}'.format(n.lat, n.lng)
        self.generateInfoLine()

    def setNewStartID(self, n):
        self.startNode = n.id
        self.generateInfoLine()

    def setNewEndID(self, n):
        self.endNode = n.id
        self.generateInfoLine()

    def updateID(self, new_id):
        self.id = new_id
        self.generateInfoLine()

    def setNewStart(self, n):
        def str2latlng(s):
            return list(map(float, s.split(' ')))
        self.startNode = n.id
        eles = self.coordLine.split(',', 2)
        twoPoints = False
        if len(eles) == 2:
            twoPoints = True
            eles = eles+['']
        startSegLen = vincenty(str2latlng(eles[0]), str2latlng(eles[1]))*1000
        newSegLen = vincenty([n.lat, n.lng], str2latlng(eles[1]))*1000
        self.length += (newSegLen-startSegLen)

        self.coordLine = '{} {},{},{}'.format(n.lat, n.lng, eles[1], eles[2])
        if twoPoints:
            # remove comma
            self.coordLine = self.coordLine[:-1]
        self.generateInfoLine()

    def setNewEnd(self, n):
        def str2latlng(s):
            return list(map(float, s.split(' ')))
        self.endNode = n.id
        eles = self.coordLine.rsplit(',', 2)
        twoPoints = False
        if len(eles) == 2:
            twoPoints = True
            eles = ['']+eles
        endSegLen = vincenty(str2latlng(eles[1]), str2latlng(eles[2]))*1000
        newSegLen = vincenty(str2latlng(eles[1]), [n.lat, n.lng])*1000
        self.length += (newSegLen-endSegLen)

        self.coordLine = '{},{},{} {}'.format(eles[0], eles[1], n.lat, n.lng)
        if twoPoints:
            # remove comma
            self.coordLine = self.coordLine[1:]
        self.generateInfoLine()

    def getStartCoord(self):
        coordStr = self.coordLine.split(',')[0]
        return (float(coordStr.split(' ')[0]), float(coordStr.split(' ')[1]))

    def getEndCoord(self):
        coordStr = self.coordLine.split(',')[-1]
        return (float(coordStr.split(' ')[0]), float(coordStr.split(' ')[1]))

    def getLength(self):
        return vincenty(self.getStartCoord(), self.getEndCoord())*1000

    def generateInfoLine(self):
        self.infoLine = str(self.id) + '\t' + str(self.startNode) + '\t' + str(self.endNode) + '\t' + \
            "{:.1f}".format(self.length) + '\t' + str(self.dir) + '\t' + str(self.maxLane) + '\t' + \
            str(self.maxSpeed) + '\t' + str(self.level) + '\t' + self.name

    def checkMerge(self, tarEdge):
        flag = 0
        if self.dir == tarEdge.dir:
            flag += 1
        if self.maxLane == tarEdge.maxLane:
            flag += 1
        if self.maxSpeed == tarEdge.maxSpeed:
            flag += 1
        if self.level == tarEdge.level:
            flag += 1
        if self.name == tarEdge.name:
            flag += 1
        if flag == 5:
            return True

    # Merge two road segment
    def merge(self, tarEdge):
        if self.endNode == tarEdge.startNode:
            # Merge in normal direction
            self.endNode = tarEdge.endNode
            self.length += tarEdge.length
            self.generateInfoLine()
            tempCoords = self.coordLine.split(',')
            tempLine = ""
            for i in range(len(tempCoords)-1):
                tempLine = tempLine + tempCoords[i] + ','
            self.coordLine = tempLine + tarEdge.coordLine
            return 1
        else:
            if self.startNode == tarEdge.endNode:
                # Merge in reverse direction
                self.startNode = tarEdge.startNode
                self.length += tarEdge.length
                self.generateInfoLine()
                tempCoords = tarEdge.coordLine.split(',')
                tempLine = ""
                for i in range(len(tempCoords) - 1):
                    tempLine = tempLine + tempCoords[i] + ','
                self.coordLine = tempLine + self.coordLine
                return -1
            else:
                return 0


# Compute the cosine value given two coordinate
def compute_angle(coord1, coord2, coord3, coord4):
    dir1 = (coord1[0]-coord2[0], coord1[1]-coord2[1])
    dir2 = (coord3[0]-coord4[0], coord3[1]-coord4[1])

    norm1 = math.sqrt(dir1[0]*dir1[0] + dir1[1]*dir1[1])
    norm2 = math.sqrt(dir2[0]*dir2[0] + dir2[1]*dir2[1])

    cosine = dir1[0]*dir2[0] + dir1[1]*dir2[1]

    return (cosine / norm1) / norm2


def load_network(inpPath):
    # Dictionary for nodes and edges
    nodes = {}
    edges = {}

    # A network for checking the degree
    G = nx.MultiGraph()

    # with open(dir + networkFileName, 'r') as netFile:   # For Python 2.7
    with open(inpPath,  'r', encoding="utf-8") as netFile:  # For Python 3
        line = netFile.readline().rstrip('\n')  # Escape the first count line
        Nnode = int(line)
        print("Number of nodes:\t" + str(Nnode))

        # Load nodes
        for i in range(Nnode):
            line = netFile.readline().rstrip('\n')
            node = GD_Node(line)
            nodes[node.id] = node
            G.add_node(node.id)

        line = netFile.readline().rstrip('\n')
        Nedge = int(line)
        print("Number of edges:\t" + str(Nedge))

        for i in range(Nedge):
            infoLine = netFile.readline().rstrip('\n')
            coordLine = netFile.readline().rstrip('\n')
            edge = GD_Edge(infoLine, coordLine)
            edges[edge.id] = edge
            G.add_edge(edge.startNode, edge.endNode, eid=edge.id)

    # import matplotlib.pyplot as plt
    # levelList=[e.level for e in edges.values()]
    # import numpy as np
    # idx, cnt=np.unique(levelList, return_counts=True)
    # plt.bar(idx, cnt)
    # plt.show()

    return nodes, edges, G


# Mode: False- only output node and edge diagnosis file; True- output simplified network as well as diagnosis file
def simplify_network(inpPath, outpPath, threshold):
    nodes, edges, G = load_network(inpPath)
    cand_nodes = []  # List of nodes with degree == 2

    Nnodes = len(nodes.keys())
    Nedges = len(edges.keys())

    # Check all nodes to find a candidate list of nodes that can be simplified
    for nid in nodes.keys():
        if G.degree[nid] == 2:
            cand_nodes.append(nid)

    Nmerge = 0
    for cnid in cand_nodes:
        # Get the edges adjacent to it
        neighbors = G.adj[cnid]
        if len(neighbors.keys()) != 2:
            continue
        eid1 = list(neighbors.items())[0][1][0]["eid"]
        eid2 = list(neighbors.items())[1][1][0]["eid"]
        # For none multigraph
        # eid1 = G[neighbors.keys()[0]][cnid]["eid"]
        # eid2 = G[neighbors.keys()[1]][cnid]["eid"]
        if eid1 < eid2:
            minId = eid1
        else:
            minId = eid2
        tarId = eid1 + eid2 - minId
        primEdge = edges[minId]
        tarEdge = edges[tarId]
        if primEdge.checkMerge(tarEdge):
            # If true, then check the direction, convex angle > threshold degree do not merge
            if abs(compute_angle(primEdge.getStartCoord(), primEdge.getEndCoord(), tarEdge.getStartCoord(), tarEdge.getEndCoord())) > threshold:
                # Edge can be merged
                outputStr = primEdge.infoLine + '\t\t' + primEdge.coordLine + \
                    '\n' + tarEdge.infoLine + '\t\t' + tarEdge.coordLine
                result = primEdge.merge(tarEdge)
                if result != 0:
                    # Merge successful, modify nodes, edges and network

                    # print("Merging segments:")
                    # print(outputStr)
                    # print(primEdge.infoLine + '\t\t' +
                    #       primEdge.coordLine + '\n')

                    G.add_edge(list(neighbors.keys())[0], list(
                        neighbors.keys())[1], eid=minId)
                    G.remove_node(cnid)
                    del nodes[cnid]
                    del edges[tarId]
                    Nnodes -= 1
                    Nedges -= 1
                    Nmerge += 1

    print("Total nodes: " + str(Nnodes) + "\tTotal edges: " +
          str(Nedges) + "\tTotal merged segments:" + str(Nmerge))

    # Output simplified network file
    with open(outpPath, 'w', encoding='utf-8') as simplifiedFile:
        # Write the nodes
        simplifiedFile.write(str(Nnodes) + '\n')
        for nId in nodes.keys():
            # outputline = nodes[nId].str + '\t' + str(G.degree[nId]) + '\n'
            outputline = nodes[nId].str + '\n'
            simplifiedFile.write(outputline)

        # Write the edges
        simplifiedFile.write(str(Nedges) + '\n')
        for eId in edges.keys():
            edge = edges[eId]
            # Check if start & end nodes in the node list
            errorLine = str(edge.id)
            if edge.startNode not in nodes.keys():
                errorLine = errorLine + \
                    "\tno startNode: " + str(edge.startNode)
                if edge.endNode not in nodes.keys():
                    errorLine = errorLine + \
                        "\tno endNode: " + str(edge.endNode)
                    print(errorLine)
                else:
                    print(errorLine)
            else:
                if edge.endNode not in nodes.keys():
                    errorLine = errorLine + \
                        "\tno endNode: " + str(edge.endNode)
                    print(errorLine)

            simplifiedFile.write(edge.infoLine + '\n')
            simplifiedFile.write(edge.coordLine + '\n')
        simplifiedFile.flush()


# Mode: `edgeLen`, group nodes by edge length; `geoDis`, gorup nodes by geo-distance.
def tf_simplify_network(inpPath, outpPath, threshold, mode='edgeLen'):
    class TFSet(object):
        def __init__(self, l):
            self.f = [i for i in range(len(l))]
            self.obj2n = {l[i]: i for i in range(len(l))}
            self.l = l.copy()

        def _find(self, x):
            if x != self.f[x]:
                self.f[x] = self._find(self.f[x])
            return self.f[x]

        def find(self, xobj):
            x = self.obj2n[xobj]
            fx = self._find(x)
            fobj = self.l[fx]
            return fobj

        def union(self, xobj, yobj):
            x, y = self.obj2n[xobj], self.obj2n[yobj]
            fx, fy = self._find(x), self._find(y)
            self.f[fy] = fx
            self.new_relationship = True

        def pin(self, xobj):
            x = self.obj2n[xobj]
            fx = self._find(x)
            self.f[x] = x
            self.f[fx] = x

        def getGroups(self, nodes):
            """
            return
                groups:
                    key: nid
                    value: group nids
                groupCeners:
                    key: nid
                    value: center of the group
            """
            if self.new_relationship:
                self.groups = dict()
                for nid in nodes:
                    fnid = self.find(nid)
                    if fnid not in self.groups:
                        self.groups[fnid] = []
                    self.groups[fnid].append(nid)
                self.groupCenters = {tmpkey: [0, 0]
                                     for tmpkey in self.groups.keys()}

                for nid, igroup in self.groups.items():
                    if len(igroup) != 0:
                        for imemb in igroup:
                            tmpNode = nodes[imemb]
                            self.groupCenters[nid][0] += tmpNode.lat
                            self.groupCenters[nid][1] += tmpNode.lng
                        self.groupCenters[nid][0] /= len(igroup)
                        self.groupCenters[nid][1] /= len(igroup)
                self.new_relationship = False
            return self.groups, self.groupCenters

    nodes, edges, G = load_network(inpPath)

    Nnodes = len(nodes.keys())
    Nedges = len(edges.keys())

    print("ORI # Nodes: {}\nORI # Edges: {}\n".format(Nnodes, Nedges))

    nodeIDList = list(nodes.keys())
    fset = TFSet(nodeIDList)

    # Union-Find to group the nodes
    for eid in edges.keys():
        e = edges[eid]
        tmpLen = e.getLength()
        if tmpLen < threshold:
            nid1, nid2 = e.startNode, e.endNode
            fset.union(nid1, nid2)

    groups, groupCenters = fset.getGroups(nodes)
    pair2dis = dict()
    pair2eid = dict()

    # select representative edge from the redundant edges
    # Merging strategy: larger level first, and shorter distance
    for eid in edges.keys():
        e = edges[eid]
        nid1, nid2 = e.startNode, e.endNode
        f1, f2 = fset.find(nid1), fset.find(nid2)
        # remove short edges
        if (f1 == f2) or e.level not in [2, 3, 4]:
            continue

        dis1 = vincenty(groupCenters[f1], [
                        nodes[nid1].lat, nodes[nid1].lng])*1000
        dis2 = vincenty(groupCenters[f2], [
                        nodes[nid2].lat, nodes[nid2].lng])*1000
        tmpdis = math.sqrt(dis1*dis1+dis2*dis2)

        if f1 < f2:
            ff1, ff2 = f1, f2
        else:
            ff1, ff2 = f2, f1
        if (ff1, ff2) not in pair2dis:
            pair2eid[(ff1, ff2)] = eid
            pair2dis[(ff1, ff2)] = tmpdis
        else:
            cur_level = edges[pair2eid[(ff1, ff2)]].level
            cur_dis = pair2dis[(ff1, ff2)]
            if (edges[eid].level == cur_level and tmpdis < cur_dis) or edges[eid].level < cur_level:
                pair2eid[(ff1, ff2)] = eid
                pair2dis[(ff1, ff2)] = tmpdis

    vstPair = list(pair2eid.keys())

    genNodeSet, genEdgeSet = set(), set()

    for ipair in vstPair:
        n1, n2 = ipair
        # append node 1
        if groupCenters[n1][0] == 0 or groupCenters[n1][1] == 0:
            raise ValueError("Union-Find error!")
        # if len(groups[n1]) != 1:
        nodes[n1].lat = groupCenters[n1][0]
        nodes[n1].lng = groupCenters[n1][1]
        nodes[n1].updateStr()
        if n1 != nodes[n1].id:
            raise ValueError("")
        if n1 not in genNodeSet:
            genNodeSet.add(n1)

        # append node 2
        if groupCenters[n2][0] == 0 or groupCenters[n2][1] == 0:
            raise ValueError("Union-Find error!")
        # if len(groups[n2]) != 1:
        nodes[n2].lat = groupCenters[n2][0]
        nodes[n2].lng = groupCenters[n2][1]
        nodes[n2].updateStr()
        if n2 != nodes[n2].id:
            raise ValueError("")
        if n2 not in genNodeSet:
            genNodeSet.add(n2)

        # append edges
        tmp_eid = pair2eid[ipair]

        tmpE = edges[tmp_eid]
        f1 = fset.find(tmpE.startNode)
        f2 = fset.find(tmpE.endNode)
        if (f1, f2) != (n1, n2) and (f2, f1) != (n1, n2):
            raise ValueError("")

        if (f1, f2) == (n1, n2):
            edges[tmp_eid].appendNewStart(nodes[n1])
            edges[tmp_eid].appendNewEnd(nodes[n2])
        else:
            edges[tmp_eid].appendNewStart(nodes[n2])
            edges[tmp_eid].appendNewEnd(nodes[n1])

        genEdgeSet.add(tmp_eid)

    Nnodes = len(genNodeSet)
    Nedges = len(genEdgeSet)
    print("Simp # Nodes: {}\nSimp # Edges: {}\n".format(Nnodes, Nedges))

    # Output simplified network file
    with open(outpPath, 'w', encoding='utf-8') as simplifiedFile:
        # Write the nodes
        simplifiedFile.write(str(Nnodes) + '\n')
        for nId in sorted(genNodeSet):
            # outputline = nodes[nId].str + '\t' + str(G.degree[nId]) + '\n'
            outputline = nodes[nId].str + '\n'
            simplifiedFile.write(outputline)

        # Write the edges
        simplifiedFile.write(str(Nedges) + '\n')
        for eId in sorted(genEdgeSet):
            edge = edges[eId]

            simplifiedFile.write(edge.infoLine + '\n')
            simplifiedFile.write(edge.coordLine + '\n')
        simplifiedFile.flush()


# def rn2json(inpPath):
#     def inChaoyang(p):
#         lat = p[0]
#         lng = p[1]
#         LB = [39.919547, 116.433453]
#         RU = [39.962322, 116.476626]
#         return LB[0] < lat < RU[0] and LB[1] < lng < RU[1]

#     nodes, edges, G = load_network(inpPath)
#     candList = []
#     candIDList = []
#     for eKey in edges.keys():
#         eles = edges[eKey].coordLine.split(',')
#         if inChaoyang(list(map(float, eles[0].split(' ')))) or inChaoyang(list(map(float, eles[1].split(' ')))):
#             candList.append(edges[eKey].coordLine)
#             candIDList.append(str(edges[eKey].id))

#     tow_template = """
# if (typeof(rn_geodata) == "undefined"){{
#     var rn_geodata={{}};
#     var rn_infostr={{}};
# }}
# rn_geodata.{0}='{1}';
# rn_infostr.{0}='{2}';
# """
#     tow = tow_template.format(rntype, ';'.join(candList), ';'.join(candIDList))
#     open(rntype+'_rndata.js', 'w').write(tow)


def formatRN(inpPath, outpPath):
    nodes, edges, G = load_network(inpPath)

    Nnodes = len(nodes)
    Nedges = len(edges)
    print("Simp # Nodes: {}\nSimp # Edges: {}\n".format(Nnodes, Nedges))

    nid2srtid = dict()
    srtNodeList = []
    for _i, nid in enumerate(nodes.keys()):
        nid2srtid[nid] = _i
        nodes[nid].updateID(_i)
        srtNodeList.append(nodes[nid])

    # Output simplified network file
    with open(outpPath, 'w', encoding='utf-8') as simplifiedFile:
        # Write the nodes
        simplifiedFile.write(str(Nnodes) + '\n')
        for _i in range(Nnodes):
            nId = srtNodeList[_i].id
            # outputline = nodes[nId].str + '\t' + str(G.degree[nId]) + '\n'
            outputline = srtNodeList[_i].str + '\n'
            simplifiedFile.write(outputline)

        # Write the edges
        simplifiedFile.write(str(Nedges) + '\n')

        _i = 0
        for eId in edges.keys():
            edge = edges[eId]

            edge.updateID(_i)
            edge.startNode = nid2srtid[edge.startNode]
            edge.endNode = nid2srtid[edge.endNode]
            edge.generateInfoLine()

            simplifiedFile.write(edge.infoLine + '\n')
            simplifiedFile.write(edge.coordLine + '\n')
            _i += 1
        simplifiedFile.flush()


def jdrn2tfrn(nodePath, edgePath, outputPath):
    import tfpkg.tfcoord

    def latlngCVT(latlngseq):
        def wgs2gcj(latlng):
            lat, lng = latlng
            retlng, retlat = tfpkg.tfcoord.wgs84_to_gcj02(lng, lat)
            return '{} {}'.format(retlat, retlng)
        import json
        latlngseq = [wgs2gcj([i['lat_wgs'], i['lng_wgs']])
                     for i in json.loads(latlngseq)]
        return ','.join(latlngseq)

    nodeTxt = open(nodePath).read().split('\n')
    # id \t lat \t lng
    tow_node = []
    for _cnt, inode in enumerate(nodeTxt):
        eles = inode.split('\t')
        nid = eles[0]
        if nid != str(_cnt):
            break
        lat, lng = float(eles[1]), float(eles[2])
        cvtLng, cvtLat = tfpkg.tfcoord.wgs84_to_gcj02(lng, lat)
        tow_node.append('{}\t{}\t{}'.format(nid, cvtLat, cvtLng))

    edgeTxt = open(edgePath, encoding='utf-8').read().split('\n')
    # infoLine: id,startNode \t endNode \t length \t dir, maxLane, maxSpeed, level, name
    # coordLine lat lng,lat lng
    tow_edge = []
    for _cnt, iedge in enumerate(edgeTxt):
        eles = iedge.split('\t')
        # eid, startNode, endNode, length, lv, name
        # [{"lat_wgs":38.86404603730905,"lng_wgs":115.34944349144075},{"lat_wgs":38.8637633897569,"lng_wgs":115.3546730682985}]
        eid = eles[0]
        if eid != str(_cnt):
            break
        startNode = eles[1]
        endNode = eles[2]
        length = eles[3]
        level = eles[4]
        name = eles[5]
        tow_edge.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            eid, startNode, endNode, length, 1, 1, 60, level, name))

        latlngseq = eles[6]
        latlngseq = latlngCVT(latlngseq)
        tow_edge.append(latlngseq)

    tow = [str(len(tow_node))]+tow_node+[str(len(tow_edge)//2)]+tow_edge
    with open(outputPath, 'w', encoding='utf-8') as outf:
        outf.write('\n'.join(tow))
        outf.write('\n')
        outf.flush()
        outf.close()


def CVT_all_in_one(cityName):
    nodePath = 'C:/data/0ori-rns/{}/node.txt'.format(cityName)
    edgePath = 'C:/data/0ori-rns/{}/edge.txt'.format(cityName)
    oriRNPath = 'C:/data/0ori-rns/{}.txt'.format(cityName)
    xysimpPath = 'C:/data/rns/{}-xianyuan.txt'.format(cityName)
    tfsimpPath = 'C:/data/rns/{}-appendNew.txt'.format(cityName)
    finalPath = 'C:/data/rns/{}.txt'.format(cityName)

    # 0ori-rns/{}/ -> 0ori-rns/{}.txt
    jdrn2tfrn(nodePath, edgePath, oriRNPath)

    # 0ori-rns/{}.txt -> rns/{}-xianyuan.txt
    threshold = 1.0/math.sqrt(3)  # angle > 60 degree do not merge
    simplify_network(oriRNPath, xysimpPath, threshold)

    # rns/{}-xianyuan.txt -> rns/{}-appendNew.txt
    threshold_merge_vertex = 30
    tf_simplify_network(xysimpPath, tfsimpPath, threshold_merge_vertex)

    # rns/{}-appendNew.txt -> rns/{}.txt
    formatRN(tfsimpPath, finalPath)

    import os
    # guo he chai qiao
    os.remove(xysimpPath)
    os.remove(tfsimpPath)


if __name__ == "__main__":
    CVT_all_in_one('baoding')
