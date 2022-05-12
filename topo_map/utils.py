import numpy as np
import matplotlib.pyplot as plt


def drawtoposkele_with_VE(graph, skeleton, v_lst, e_lst, ax=plt):
	h, w = skeleton.shape
	res = skeleton
	ax.imshow(res, cmap='GnBu')

	x, y = [], []
	for ed in e_lst:
		v1 = v_lst[ed[0]]
		v2 = v_lst[ed[1]]
		y.append(v1[1])
		x.append(v1[0])
		y.append(v2[1])
		x.append(v2[0])
		ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 
                'k-', lw=1)
	ax.scatter(x=x, y=y, c='r', s=2) 


def build_VE_from_graph (graph, skeleton, vertex_dist=5):
	'''
	vectex_dist is counted in pixels
	'''

	#store the vertex location
	h, w = skeleton.shape
	v_lst = []
	e_lst = []

	def exist_in_list(v_in):
		for i, v in enumerate(v_lst):
			dist = np.sum((v - v_in)**2)
			if dist < vertex_dist**2:
				return True, i
		return False, len(v_lst)

	for ed in graph.edges:
		if ed.parentEdgeId == -1:
			#print('ed.vertices = {}'.format(ed.vertices))

			len_edge = len(ed.vertices)
			for idx_v in range(len_edge-1):
				v1 = np.array((ed.vertices[idx_v] % w, ed.vertices[idx_v] // w))
				v2 = np.array((ed.vertices[idx_v+1] % w, ed.vertices[idx_v+1] // w))
				#assert 1==2
				flag_v1, idx_v1 = exist_in_list(v1)
				if not flag_v1:
					#print('append v1 ...')
					v_lst.append(v1)
				flag_v2, idx_v2 = exist_in_list(v2)
				if not flag_v2:
					#print('append v2 ...')
					v_lst.append(v2)
				if idx_v1 != idx_v2:
					e_lst.append((idx_v1, idx_v2))
					#print('add edge ({}, {})'.format(idx_v1, idx_v2))
				#print('---------------------------------------------------------')
	return v_lst, e_lst