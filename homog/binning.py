import numpy as np
import traitlets as tl


class PositiveInt(tl.Int):

    def info(self):
        return u'a positive integer'

    def validate(self, obj, proposal):
        super().validate(obj, proposal)
        if proposal <= 0:
            self.error(obj, proposal)
        return proposal


class PositiveFloat(tl.Float):

    def info(self):
        return u'a positive float'

    def validate(self, obj, proposal):
        super().validate(obj, proposal)
        if proposal <= 0.0:
            self.error(obj, proposal)
        return proposal


_R22 = np.sqrt(2) / 2
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
    [_R22, _R22, 0, 0],
    [_R22, 0, _R22, 0],
    [_R22, 0, 0, _R22],
    [0, _R22, _R22, 0],
    [0, _R22, 0, _R22],
    [0, 0, _R22, _R22],
    [-_R22, _R22, 0, 0],
    [-_R22, 0, _R22, 0],
    [-_R22, 0, 0, _R22],
    [0, -_R22, _R22, 0],
    [0, -_R22, 0, _R22],
    [0, 0, -_R22, _R22]
])


def half48cell_face(quat):
    # maybe faster to do the c++ stuff:
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
        # Index facecell = hyperface_axis;  // | (quat[hyperface_axis]<0 ? 4 :
        # 0);

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
    quat = np.asarray(quat)
    fullaxes = (slice(None),) + (np.newaxis,) * (quat.ndim - 1)
    hf = half48cell_faces[fullaxes]
    if quat.ndim > 1:
        print(hf.shape)
        print(quat[np.newaxis].shape)
    dots = abs(np.sum(quat[np.newaxis] * hf, axis=-1))
    return np.argmax(dots, axis=0)


_xform_binner_covrad = np.array([
    49.66580, 25.99805, 17.48845, 13.15078, 10.48384, 8.76800, 7.48210,
    6.56491, 5.84498, 5.27430, 4.78793, 4.35932, 4.04326, 3.76735,
    3.51456, 3.29493, 3.09656, 2.92407, 2.75865, 2.62890, 2.51173,
    2.39665, 2.28840, 2.19235, 2.09949, 2.01564, 1.94154, 1.87351,
    1.80926, 1.75516, 1.69866, 1.64672, 1.59025, 1.54589, 1.50077,
    1.46216, 1.41758, 1.38146, 1.35363, 1.31630, 1.28212, 1.24864,
    1.21919, 1.20169, 1.17003, 1.14951, 1.11853, 1.09436, 1.07381,
    1.05223, 1.02896, 1.00747, 0.99457, 0.97719, 0.95703, 0.93588,
    0.92061, 0.90475, 0.89253, 0.87480, 0.86141, 0.84846, 0.83677,
    0.82164])


class XformBinner(tl.HasTraits):
    cart_resl = PositiveFloat(default_value=1)
    ori_resl = PositiveFloat(default_value=15)
    cart_bound = PositiveFloat(default_value=512)
    ori_nside = PositiveInt()

    @tl.default('ori_nside')
    def _default_ori_nside(self):
        return int(np.sum(_xform_binner_covrad >= self.ori_resl) + 1)
