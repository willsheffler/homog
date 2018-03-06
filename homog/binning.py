import numpy as np


# template <class V4, class Index>
# void get_cell_48cell_half(V4 const &quat, Index &cell) {
#   typedef typename V4::Scalar Float;
#   V4 const quat_pos = quat.cwiseAbs();
#   V4 tmpv = quat_pos;

#   Float hyperface_dist;  // dist to closest face
#   Index hyperface_axis;  // closest hyperface-pair
#   Float edge_dist;       // dist to closest edge
#   Index edge_axis_1;     // first axis of closest edge
#   Index edge_axis_2;     // second axis of closest edge
#   Float corner_dist;     // dist to closest corner

#   // std::cout << quat_pos.transpose() << std::endl;
#   numeric::max2(quat_pos, hyperface_dist, edge_dist, hyperface_axis,
#                 edge_axis_2);
#   edge_dist = sqrt(2) / 2 * (hyperface_dist + edge_dist);
#   corner_dist = quat_pos.sum() / 2;
#   // std::cout << hyperface_axis << " " << edge_axis_2 << std::endl;
#   edge_axis_1 = hyperface_axis < edge_axis_2 ? hyperface_axis : edge_axis_2;
#   edge_axis_2 = hyperface_axis < edge_axis_2 ? edge_axis_2 : hyperface_axis;
#   assert(edge_axis_1 < edge_axis_2);

#   // cell if closest if of form 1000 (add 4 if negative)
#   Index facecell = hyperface_axis;  // | (quat[hyperface_axis]<0 ? 4 : 0);

#   // cell if closest is of form 1111, bitwise by ( < 0)
#   Index bit0 = quat[0] < 0;
#   Index bit1 = quat[1] < 0;
#   Index bit2 = quat[2] < 0;
#   Index cornercell = quat[3] > 0 ? bit0 | bit1 << 1 | bit2 << 2
#                                  : (!bit0) | (!bit1) << 1 | (!bit2) << 2;

#   // cell if closest is of form 1100
#   Index perm_shift[3][4] = {{9, 0, 1, 2}, {0, 9, 3, 4}, {1, 3, 9, 5}};
#   Index sign_shift = (quat[edge_axis_1] < 0 != quat[edge_axis_2] < 0) * 1 * 6;
#   Index edgecell = sign_shift + perm_shift[edge_axis_1][edge_axis_2];

#   // pick case 1000 1111 1100 without if statements
#   Index swtch;
#   util::SimpleArray<3, Float>(hyperface_dist, corner_dist, edge_dist)
#       .maxCoeff(&swtch);
#   cell = swtch == 0 ? facecell : (swtch == 1 ? cornercell + 4 : edgecell + 12);
#   // this is slower !?!
#   // Float mx = std::max(std::max(hyperface_dist,corner_dist),edge_dist);
#   // cell2[i] = hyperface_dist==mx ? facecell : (corner_dist==mx ? cornercell+8
#   // : edgecell+24);
# }

half48cell_faces = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0.5, 0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5, 0.5],
    [np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0],
    [np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
    [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
    [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
    [0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
    [0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2],
    [-np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0],
    [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0],
    [-np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
    [0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
    [0, -np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
    [0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2]
])


def half48cell_face(quat):
    # maybe faster to do the above c++ stuff
    quat = np.asarray(quat)
    fullaxes = (slice(None),) + (np.newaxis,) * (quat.ndim - 1)
    hf = half48cell_faces[fullaxes]
    if quat.ndim > 1:
        print(hf.shape)
        print(quat[np.newaxis].shape)
    dots = abs(np.sum(quat[np.newaxis] * hf, axis=-1))
    return np.argmax(dots, axis=0)
