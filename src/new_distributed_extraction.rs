use crate::pyramid::*;
use image::Rgb;
use image::{GenericImageView, GrayImage};
use imageproc::corners::corners_fast9;
use imageproc::drawing::draw_cross_mut;

// None of these parameters are tuned by me, taken from ORB-SLAM2 directory
const EDGE_THRESHOLD: i32 = 19;
const PATCH_SIZE: i32 = 31;
const HALF_PATCH_SIZE: i32 = 15;

#[derive(Debug, Clone)]
pub struct OrbDescriptor {
    pub keypoint: KeyPoint,
    pub descriptor: Vec<u8>, // 32 bytes for 256-bit descriptor
}

#[derive(Debug, Clone)]
pub struct KeyPoint {
    pub pt: (f32, f32),
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

#[derive(Debug, Clone)]
struct ExtractorNode {
    ul: (i32, i32),
    ur: (i32, i32),
    bl: (i32, i32),
    br: (i32, i32),
    keypoints: Vec<KeyPoint>,
    no_more: bool,
}

const BIT_PATTERN_31: [(i32, i32); 512] = [
    (8, -3),
    (9, 5), // mean (0), correlation (0)
    (4, 2),
    (7, -12), // mean (1.12461e-05), correlation (0.0437584)
    (-11, 9),
    (-8, 2), // mean (3.37382e-05), correlation (0.0617409)
    (7, -12),
    (12, -13), // mean (5.62303e-05), correlation (0.0636977)
    (2, -13),
    (2, 12), // mean (0.000134953), correlation (0.085099)
    (1, -7),
    (1, 6), // mean (0.000528565), correlation (0.0857175)
    (-2, -10),
    (-2, -4), // mean (0.0188821), correlation (0.0985774)
    (-13, -13),
    (-11, -8), // mean (0.0363135), correlation (0.0899616)
    (-13, -3),
    (-12, -9), // mean (0.121806), correlation (0.099849)
    (10, 4),
    (11, 9), // mean (0.122065), correlation (0.093285)
    (-13, -8),
    (-8, -9), // mean (0.162787), correlation (0.0942748)
    (-11, 7),
    (-9, 12), // mean (0.21561), correlation (0.0974438)
    (7, 7),
    (12, 6), // mean (0.160583), correlation (0.130064)
    (-4, -5),
    (-3, 0), // mean (0.228171), correlation (0.132998)
    (-13, 2),
    (-12, -3), // mean (0.00997526), correlation (0.145926)
    (-9, 0),
    (-7, 5), // mean (0.198234), correlation (0.143636)
    (12, -6),
    (12, -1), // mean (0.0676226), correlation (0.16689)
    (-3, 6),
    (-2, 12), // mean (0.166847), correlation (0.171682)
    (-6, -13),
    (-4, -8), // mean (0.101215), correlation (0.179716)
    (11, -13),
    (12, -8), // mean (0.200641), correlation (0.192279)
    (4, 7),
    (5, 1), // mean (0.205106), correlation (0.186848)
    (5, -3),
    (10, -3), // mean (0.234908), correlation (0.192319)
    (3, -7),
    (6, 12), // mean (0.0709964), correlation (0.210872)
    (-8, -7),
    (-6, -2), // mean (0.0939834), correlation (0.212589)
    (-2, 11),
    (-1, -10), // mean (0.127778), correlation (0.20866)
    (-13, 12),
    (-8, 10), // mean (0.14783), correlation (0.206356)
    (-7, 3),
    (-5, -3), // mean (0.182141), correlation (0.198942)
    (-4, 2),
    (-3, 7), // mean (0.188237), correlation (0.21384)
    (-10, -12),
    (-6, 11), // mean (0.14865), correlation (0.23571)
    (5, -12),
    (6, -7), // mean (0.222312), correlation (0.23324)
    (5, -6),
    (7, -1), // mean (0.229082), correlation (0.23389)
    (1, 0),
    (4, -5), // mean (0.241577), correlation (0.215286)
    (9, 11),
    (11, -13), // mean (0.00338507), correlation (0.251373)
    (4, 7),
    (4, 12), // mean (0.131005), correlation (0.257622)
    (2, -1),
    (4, 4), // mean (0.152755), correlation (0.255205)
    (-4, -12),
    (-2, 7), // mean (0.182771), correlation (0.244867)
    (-8, -5),
    (-7, -10), // mean (0.186898), correlation (0.23901)
    (4, 11),
    (9, 12), // mean (0.226226), correlation (0.258255)
    (0, -8),
    (1, -13), // mean (0.0897886), correlation (0.274827)
    (-13, -2),
    (-8, 2), // mean (0.148774), correlation (0.28065)
    (-3, -2),
    (-2, 3), // mean (0.153048), correlation (0.283063)
    (-6, 9),
    (-4, -9), // mean (0.169523), correlation (0.278248)
    (8, 12),
    (10, 7), // mean (0.225337), correlation (0.282851)
    (0, 9),
    (1, 3), // mean (0.226687), correlation (0.278734)
    (7, -5),
    (11, -10), // mean (0.00693882), correlation (0.305161)
    (-13, -6),
    (-11, 0), // mean (0.0227283), correlation (0.300181)
    (10, 7),
    (12, 1), // mean (0.125517), correlation (0.31089)
    (-6, -3),
    (-6, 12), // mean (0.131748), correlation (0.312779)
    (10, -9),
    (12, -4), // mean (0.144827), correlation (0.292797)
    (-13, 8),
    (-8, -12), // mean (0.149202), correlation (0.308918)
    (-13, 0),
    (-8, -4), // mean (0.160909), correlation (0.310013)
    (3, 3),
    (7, 8), // mean (0.177755), correlation (0.309394)
    (5, 7),
    (10, -7), // mean (0.212337), correlation (0.310315)
    (-1, 7),
    (1, -12), // mean (0.214429), correlation (0.311933)
    (3, -10),
    (5, 6), // mean (0.235807), correlation (0.313104)
    (2, -4),
    (3, -10), // mean (0.00494827), correlation (0.344948)
    (-13, 0),
    (-13, 5), // mean (0.0549145), correlation (0.344675)
    (-13, -7),
    (-12, 12), // mean (0.103385), correlation (0.342715)
    (-13, 3),
    (-11, 8), // mean (0.134222), correlation (0.322922)
    (-7, 12),
    (-4, 7), // mean (0.153284), correlation (0.337061)
    (6, -10),
    (12, 8), // mean (0.154881), correlation (0.329257)
    (-9, -1),
    (-7, -6), // mean (0.200967), correlation (0.33312)
    (-2, -5),
    (0, 12), // mean (0.201518), correlation (0.340635)
    (-12, 5),
    (-7, 5), // mean (0.207805), correlation (0.335631)
    (3, -10),
    (8, -13), // mean (0.224438), correlation (0.34504)
    (-7, -7),
    (-4, 5), // mean (0.239361), correlation (0.338053)
    (-3, -2),
    (-1, -7), // mean (0.240744), correlation (0.344322)
    (2, 9),
    (5, -11), // mean (0.242949), correlation (0.34145)
    (-11, -13),
    (-5, -13), // mean (0.244028), correlation (0.336861)
    (-1, 6),
    (0, -1), // mean (0.247571), correlation (0.343684)
    (5, -3),
    (5, 2), // mean (0.000697256), correlation (0.357265)
    (-4, -13),
    (-4, 12), // mean (0.00213675), correlation (0.373827)
    (-9, -6),
    (-9, 6), // mean (0.0126856), correlation (0.373938)
    (-12, -10),
    (-8, -4), // mean (0.0152497), correlation (0.364237)
    (10, 2),
    (12, -3), // mean (0.0299933), correlation (0.345292)
    (7, 12),
    (12, 12), // mean (0.0307242), correlation (0.366299)
    (-7, -13),
    (-6, 5), // mean (0.0534975), correlation (0.368357)
    (-4, 9),
    (-3, 4), // mean (0.099865), correlation (0.372276)
    (7, -1),
    (12, 2), // mean (0.117083), correlation (0.364529)
    (-7, 6),
    (-5, 1), // mean (0.126125), correlation (0.369606)
    (-13, 11),
    (-12, 5), // mean (0.130364), correlation (0.358502)
    (-3, 7),
    (-2, -6), // mean (0.131691), correlation (0.375531)
    (7, -8),
    (12, -7), // mean (0.160166), correlation (0.379508)
    (-13, -7),
    (-11, -12), // mean (0.167848), correlation (0.353343)
    (1, -3),
    (12, 12), // mean (0.183378), correlation (0.371916)
    (2, -6),
    (3, 0), // mean (0.228711), correlation (0.371761)
    (-4, 3),
    (-2, -13), // mean (0.247211), correlation (0.364063)
    (-1, -13),
    (1, 9), // mean (0.249325), correlation (0.378139)
    (7, 1),
    (8, -6), // mean (0.000652272), correlation (0.411682)
    (1, -1),
    (3, 12), // mean (0.00248538), correlation (0.392988)
    (9, 1),
    (12, 6), // mean (0.0206815), correlation (0.386106)
    (-1, -9),
    (-1, 3), // mean (0.0364485), correlation (0.410752)
    (-13, -13),
    (-10, 5), // mean (0.0376068), correlation (0.398374)
    (7, 7),
    (10, 12), // mean (0.0424202), correlation (0.405663)
    (12, -5),
    (12, 9), // mean (0.0942645), correlation (0.410422)
    (6, 3),
    (7, 11), // mean (0.1074), correlation (0.413224)
    (5, -13),
    (6, 10), // mean (0.109256), correlation (0.408646)
    (2, -12),
    (2, 3), // mean (0.131691), correlation (0.416076)
    (3, 8),
    (4, -6), // mean (0.165081), correlation (0.417569)
    (2, 6),
    (12, -13), // mean (0.171874), correlation (0.408471)
    (9, -12),
    (10, 3), // mean (0.175146), correlation (0.41296)
    (-8, 4),
    (-7, 9), // mean (0.183682), correlation (0.402956)
    (-11, 12),
    (-4, -6), // mean (0.184672), correlation (0.416125)
    (1, 12),
    (2, -8), // mean (0.191487), correlation (0.386696)
    (6, -9),
    (7, -4), // mean (0.192668), correlation (0.394771)
    (2, 3),
    (3, -2), // mean (0.200157), correlation (0.408303)
    (6, 3),
    (11, 0), // mean (0.204588), correlation (0.411762)
    (3, -3),
    (8, -8), // mean (0.205904), correlation (0.416294)
    (7, 8),
    (9, 3), // mean (0.213237), correlation (0.409306)
    (-11, -5),
    (-6, -4), // mean (0.243444), correlation (0.395069)
    (-10, 11),
    (-5, 10), // mean (0.247672), correlation (0.413392)
    (-5, -8),
    (-3, 12), // mean (0.24774), correlation (0.411416)
    (-10, 5),
    (-9, 0), // mean (0.00213675), correlation (0.454003)
    (8, -1),
    (12, -6), // mean (0.0293635), correlation (0.455368)
    (4, -6),
    (6, -11), // mean (0.0404971), correlation (0.457393)
    (-10, 12),
    (-8, 7), // mean (0.0481107), correlation (0.448364)
    (4, -2),
    (6, 7), // mean (0.050641), correlation (0.455019)
    (-2, 0),
    (-2, 12), // mean (0.0525978), correlation (0.44338)
    (-5, -8),
    (-5, 2), // mean (0.0629667), correlation (0.457096)
    (7, -6),
    (10, 12), // mean (0.0653846), correlation (0.445623)
    (-9, -13),
    (-8, -8), // mean (0.0858749), correlation (0.449789)
    (-5, -13),
    (-5, -2), // mean (0.122402), correlation (0.450201)
    (8, -8),
    (9, -13), // mean (0.125416), correlation (0.453224)
    (-9, -11),
    (-9, 0), // mean (0.130128), correlation (0.458724)
    (1, -8),
    (1, -2), // mean (0.132467), correlation (0.440133)
    (7, -4),
    (9, 1), // mean (0.132692), correlation (0.454)
    (-2, 1),
    (-1, -4), // mean (0.135695), correlation (0.455739)
    (11, -6),
    (12, -11), // mean (0.142904), correlation (0.446114)
    (-12, -9),
    (-6, 4), // mean (0.146165), correlation (0.451473)
    (3, 7),
    (7, 12), // mean (0.147627), correlation (0.456643)
    (5, 5),
    (10, 8), // mean (0.152901), correlation (0.455036)
    (0, -4),
    (2, 8), // mean (0.167083), correlation (0.459315)
    (-9, 12),
    (-5, -13), // mean (0.173234), correlation (0.454706)
    (0, 7),
    (2, 12), // mean (0.18312), correlation (0.433855)
    (-1, 2),
    (1, 7), // mean (0.185504), correlation (0.443838)
    (5, 11),
    (7, -9), // mean (0.185706), correlation (0.451123)
    (3, 5),
    (6, -8), // mean (0.188968), correlation (0.455808)
    (-13, -4),
    (-8, 9), // mean (0.191667), correlation (0.459128)
    (-5, 9),
    (-3, -3), // mean (0.193196), correlation (0.458364)
    (-4, -7),
    (-3, -12), // mean (0.196536), correlation (0.455782)
    (6, 5),
    (8, 0), // mean (0.1972), correlation (0.450481)
    (-7, 6),
    (-6, 12), // mean (0.199438), correlation (0.458156)
    (-13, 6),
    (-5, -2), // mean (0.211224), correlation (0.449548)
    (1, -10),
    (3, 10), // mean (0.211718), correlation (0.440606)
    (4, 1),
    (8, -4), // mean (0.213034), correlation (0.443177)
    (-2, -2),
    (2, -13), // mean (0.234334), correlation (0.455304)
    (2, -12),
    (12, 12), // mean (0.235684), correlation (0.443436)
    (-2, -13),
    (0, -6), // mean (0.237674), correlation (0.452525)
    (4, 1),
    (9, 3), // mean (0.23962), correlation (0.444824)
    (-6, -10),
    (-3, -5), // mean (0.248459), correlation (0.439621)
    (-3, -13),
    (-1, 1), // mean (0.249505), correlation (0.456666)
    (7, 5),
    (12, -11), // mean (0.00119208), correlation (0.495466)
    (4, -2),
    (5, -7), // mean (0.00372245), correlation (0.484214)
    (-13, 9),
    (-9, -5), // mean (0.00741116), correlation (0.499854)
    (7, 1),
    (8, 6), // mean (0.0208952), correlation (0.499773)
    (7, -8),
    (7, 6), // mean (0.0220085), correlation (0.501609)
    (-7, -4),
    (-7, 1), // mean (0.0233806), correlation (0.496568)
    (-8, 11),
    (-7, -8), // mean (0.0236505), correlation (0.489719)
    (-13, 6),
    (-12, -8), // mean (0.0268781), correlation (0.503487)
    (2, 4),
    (3, 9), // mean (0.0323324), correlation (0.501938)
    (10, -5),
    (12, 3), // mean (0.0399235), correlation (0.494029)
    (-6, -5),
    (-6, 7), // mean (0.0420153), correlation (0.486579)
    (8, -3),
    (9, -8), // mean (0.0548021), correlation (0.484237)
    (2, -12),
    (2, 8), // mean (0.0616622), correlation (0.496642)
    (-11, -2),
    (-10, 3), // mean (0.0627755), correlation (0.498563)
    (-12, -13),
    (-7, -9), // mean (0.0829622), correlation (0.495491)
    (-11, 0),
    (-10, -5), // mean (0.0843342), correlation (0.487146)
    (5, -3),
    (11, 8), // mean (0.0929937), correlation (0.502315)
    (-2, -13),
    (-1, 12), // mean (0.113327), correlation (0.48941)
    (-1, -8),
    (0, 9), // mean (0.132119), correlation (0.467268)
    (-13, -11),
    (-12, -5), // mean (0.136269), correlation (0.498771)
    (-10, -2),
    (-10, 11), // mean (0.142173), correlation (0.498714)
    (-3, 9),
    (-2, -13), // mean (0.144141), correlation (0.491973)
    (2, -3),
    (3, 2), // mean (0.14892), correlation (0.500782)
    (-9, -13),
    (-4, 0), // mean (0.150371), correlation (0.498211)
    (-4, 6),
    (-3, -10), // mean (0.152159), correlation (0.495547)
    (-4, 12),
    (-2, -7), // mean (0.156152), correlation (0.496925)
    (-6, -11),
    (-4, 9), // mean (0.15749), correlation (0.499222)
    (6, -3),
    (6, 11), // mean (0.159211), correlation (0.503821)
    (-13, 11),
    (-5, 5), // mean (0.162427), correlation (0.501907)
    (11, 11),
    (12, 6), // mean (0.16652), correlation (0.497632)
    (7, -5),
    (12, -2), // mean (0.169141), correlation (0.484474)
    (-1, 12),
    (0, 7), // mean (0.169456), correlation (0.495339)
    (-4, -8),
    (-3, -2), // mean (0.171457), correlation (0.487251)
    (-7, 1),
    (-6, 7), // mean (0.175), correlation (0.500024)
    (-13, -12),
    (-8, -13), // mean (0.175866), correlation (0.497523)
    (-7, -2),
    (-6, -8), // mean (0.178273), correlation (0.501854)
    (-8, 5),
    (-6, -9), // mean (0.181107), correlation (0.494888)
    (-5, -1),
    (-4, 5), // mean (0.190227), correlation (0.482557)
    (-13, 7),
    (-8, 10), // mean (0.196739), correlation (0.496503)
    (1, 5),
    (5, -13), // mean (0.19973), correlation (0.499759)
    (1, 0),
    (10, -13), // mean (0.204465), correlation (0.49873)
    (9, 12),
    (10, -1), // mean (0.209334), correlation (0.49063)
    (5, -8),
    (10, -9), // mean (0.211134), correlation (0.503011)
    (-1, 11),
    (1, -13), // mean (0.212), correlation (0.499414)
    (-9, -3),
    (-6, 2), // mean (0.212168), correlation (0.480739)
    (-1, -10),
    (1, 12), // mean (0.212731), correlation (0.502523)
    (-13, 1),
    (-8, -10), // mean (0.21327), correlation (0.489786)
    (8, -11),
    (10, -6), // mean (0.214159), correlation (0.488246)
    (2, -13),
    (3, -6), // mean (0.216993), correlation (0.50287)
    (7, -13),
    (12, -9), // mean (0.223639), correlation (0.470502)
    (-10, -10),
    (-5, -7), // mean (0.224089), correlation (0.500852)
    (-10, -8),
    (-8, -13), // mean (0.228666), correlation (0.502629)
    (4, -6),
    (8, 5), // mean (0.22906), correlation (0.498305)
    (3, 12),
    (8, -13), // mean (0.233378), correlation (0.503825)
    (-4, 2),
    (-3, -3), // mean (0.234323), correlation (0.476692)
    (5, -13),
    (10, -12), // mean (0.236392), correlation (0.475462)
    (4, -13),
    (5, -1), // mean (0.236842), correlation (0.504132)
    (-9, 9),
    (-4, 3), // mean (0.236977), correlation (0.497739)
    (0, 3),
    (3, -9), // mean (0.24314), correlation (0.499398)
    (-12, 1),
    (-6, 1), // mean (0.243297), correlation (0.489447)
    (3, 2),
    (4, -8), // mean (0.00155196), correlation (0.553496)
    (-10, -10),
    (-10, 9), // mean (0.00239541), correlation (0.54297)
    (8, -13),
    (12, 12), // mean (0.0034413), correlation (0.544361)
    (-8, -12),
    (-6, -5), // mean (0.003565), correlation (0.551225)
    (2, 2),
    (3, 7), // mean (0.00835583), correlation (0.55285)
    (10, 6),
    (11, -8), // mean (0.00885065), correlation (0.540913)
    (6, 8),
    (8, -12), // mean (0.0101552), correlation (0.551085)
    (-7, 10),
    (-6, 5), // mean (0.0102227), correlation (0.533635)
    (-3, -9),
    (-3, 9), // mean (0.0110211), correlation (0.543121)
    (-1, -13),
    (-1, 5), // mean (0.0113473), correlation (0.550173)
    (-3, -7),
    (-3, 4), // mean (0.0140913), correlation (0.554774)
    (-8, -2),
    (-8, 3), // mean (0.017049), correlation (0.55461)
    (4, 2),
    (12, 12), // mean (0.01778), correlation (0.546921)
    (2, -5),
    (3, 11), // mean (0.0224022), correlation (0.549667)
    (6, -9),
    (11, -13), // mean (0.029161), correlation (0.546295)
    (3, -1),
    (7, 12), // mean (0.0303081), correlation (0.548599)
    (11, -1),
    (12, 4), // mean (0.0355151), correlation (0.523943)
    (-3, 0),
    (-3, 6), // mean (0.0417904), correlation (0.543395)
    (4, -11),
    (4, 12), // mean (0.0487292), correlation (0.542818)
    (2, -4),
    (2, 1), // mean (0.0575124), correlation (0.554888)
    (-10, -6),
    (-8, 1), // mean (0.0594242), correlation (0.544026)
    (-13, 7),
    (-11, 1), // mean (0.0597391), correlation (0.550524)
    (-13, 12),
    (-11, -13), // mean (0.0608974), correlation (0.55383)
    (6, 0),
    (11, -13), // mean (0.065126), correlation (0.552006)
    (0, -1),
    (1, 4), // mean (0.074224), correlation (0.546372)
    (-13, 3),
    (-9, -2), // mean (0.0808592), correlation (0.554875)
    (-9, 8),
    (-6, -3), // mean (0.0883378), correlation (0.551178)
    (-13, -6),
    (-8, -2), // mean (0.0901035), correlation (0.548446)
    (5, -9),
    (8, 10), // mean (0.0949843), correlation (0.554694)
    (2, 7),
    (3, -9), // mean (0.0994152), correlation (0.550979)
    (-1, -6),
    (-1, -1), // mean (0.10045), correlation (0.552714)
    (9, 5),
    (11, -2), // mean (0.100686), correlation (0.552594)
    (11, -3),
    (12, -8), // mean (0.101091), correlation (0.532394)
    (3, 0),
    (3, 5), // mean (0.101147), correlation (0.525576)
    (-1, 4),
    (0, 10), // mean (0.105263), correlation (0.531498)
    (3, -6),
    (4, 5), // mean (0.110785), correlation (0.540491)
    (-13, 0),
    (-10, 5), // mean (0.112798), correlation (0.536582)
    (5, 8),
    (12, 11), // mean (0.114181), correlation (0.555793)
    (8, 9),
    (9, -6), // mean (0.117431), correlation (0.553763)
    (7, -4),
    (8, -12), // mean (0.118522), correlation (0.553452)
    (-10, 4),
    (-10, 9), // mean (0.12094), correlation (0.554785)
    (7, 3),
    (12, 4), // mean (0.122582), correlation (0.555825)
    (9, -7),
    (10, -2), // mean (0.124978), correlation (0.549846)
    (7, 0),
    (12, -2), // mean (0.127002), correlation (0.537452)
    (-1, -6),
    (0, -11), // mean (0.127148), correlation (0.547401)
];

fn compute_orb_descriptor(image: &GrayImage, kp: &KeyPoint) -> Vec<u8> {
    let mut descriptor = vec![0u8; 32]; // 256 bits = 32 bytes
    let angle = kp.angle * std::f32::consts::PI / 180.0;
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    let x = kp.pt.0.round() as i32;
    let y = kp.pt.1.round() as i32;
    let (width, height) = image.dimensions();

    // Process 16 points at a time to build each byte
    for i in 0..32 {
        let mut val: u8 = 0;

        // Compare 8 point pairs to create each byte
        for j in 0..8 {
            let pattern_idx = i * 8 + j;
            let (x1, y1) = BIT_PATTERN_31[pattern_idx * 2];
            let (x2, y2) = BIT_PATTERN_31[pattern_idx * 2 + 1];

            // Rotate the pattern points according to keypoint orientation
            let rx1 = (cos_angle * x1 as f32 - sin_angle * y1 as f32).round() as i32;
            let ry1 = (sin_angle * x1 as f32 + cos_angle * y1 as f32).round() as i32;
            let rx2 = (cos_angle * x2 as f32 - sin_angle * y2 as f32).round() as i32;
            let ry2 = (sin_angle * x2 as f32 + cos_angle * y2 as f32).round() as i32;

            let px1 = (x + rx1).clamp(0, width as i32 - 1) as u32;
            let py1 = (y + ry1).clamp(0, height as i32 - 1) as u32;
            let px2 = (x + rx2).clamp(0, width as i32 - 1) as u32;
            let py2 = (y + ry2).clamp(0, height as i32 - 1) as u32;

            let p1 = image.get_pixel(px1, py1).0[0];
            let p2 = image.get_pixel(px2, py2).0[0];

            // Set bit based on intensity comparison
            if p1 < p2 {
                val |= 1 << j;
            }
        }
        descriptor[i] = val;
    }

    descriptor
}

impl ExtractorNode {
    fn new(ul: (i32, i32), ur: (i32, i32), bl: (i32, i32), br: (i32, i32)) -> Self {
        ExtractorNode {
            ul,
            ur,
            bl,
            br,
            keypoints: Vec::new(),
            no_more: false,
        }
    }

    fn divide(&self) -> (ExtractorNode, ExtractorNode, ExtractorNode, ExtractorNode) {
        let half_x = ((self.ur.0 - self.ul.0) as f32 / 2.0).ceil() as i32;
        let half_y = ((self.bl.1 - self.ul.1) as f32 / 2.0).ceil() as i32;

        let n1 = ExtractorNode::new(
            self.ul,
            (self.ul.0 + half_x, self.ul.1),
            (self.ul.0, self.ul.1 + half_y),
            (self.ul.0 + half_x, self.ul.1 + half_y),
        );

        let n2 = ExtractorNode::new(
            (self.ul.0 + half_x, self.ul.1),
            self.ur,
            (self.ul.0 + half_x, self.ul.1 + half_y),
            (self.ur.0, self.ul.1 + half_y),
        );

        let n3 = ExtractorNode::new(
            (self.ul.0, self.ul.1 + half_y),
            (self.ul.0 + half_x, self.ul.1 + half_y),
            self.bl,
            (self.bl.0 + half_x, self.bl.1),
        );

        let n4 = ExtractorNode::new(
            (self.ul.0 + half_x, self.ul.1 + half_y),
            (self.ur.0, self.ul.1 + half_y),
            (self.bl.0 + half_x, self.bl.1),
            self.br,
        );

        (n1, n2, n3, n4)
    }
}

// this just closely follows the ORB-SLAM2 orientation computation
fn compute_orientation(image: &GrayImage, keypoint: &mut KeyPoint) -> f32 {
    let x = keypoint.pt.0.round() as i32;
    let y = keypoint.pt.1.round() as i32;
    let (width, height) = image.dimensions();

    let mut m_01 = 0i32;
    let mut m_10 = 0i32;

    for u in -HALF_PATCH_SIZE..=HALF_PATCH_SIZE {
        let x_coord = (x + u).clamp(0, width as i32 - 1) as u32;
        m_10 += u * image.get_pixel(x_coord, y as u32).0[0] as i32;
    }

    for v in 1..=HALF_PATCH_SIZE {
        // Compute max u for this v value using a circular patch
        let u_max = ((HALF_PATCH_SIZE * HALF_PATCH_SIZE - v * v) as f32).sqrt() as i32;

        for u in -u_max..=u_max {
            let x_coord = (x + u).clamp(0, width as i32 - 1) as u32;
            let y_plus = (y + v).clamp(0, height as i32 - 1) as u32;
            let y_minus = (y - v).clamp(0, height as i32 - 1) as u32;

            let val_plus = image.get_pixel(x_coord, y_plus).0[0] as i32;
            let val_minus = image.get_pixel(x_coord, y_minus).0[0] as i32;

            m_10 += u * (val_plus + val_minus);
            m_01 += v * (val_plus - val_minus);
        }
    }

    (m_01 as f32).atan2(m_10 as f32) * 180.0 / std::f32::consts::PI
}

pub fn compute_descriptor_quad_tree(
    pyramid: &ImagePyramid,
    ini_th_fast: u8,
    min_th_fast: u8,
    max_keypoints: usize,
) -> Vec<Vec<OrbDescriptor>> {
    let mut all_descriptors = Vec::new();
    let w: i32 = 30; // Cell size

    for (level, image) in pyramid.levels.iter().enumerate() {
        let (width, height) = image.dimensions();

        let min_border_x = EDGE_THRESHOLD - 3;
        let max_border_x = width as i32 - EDGE_THRESHOLD + 3;
        let min_border_y = min_border_x;
        let max_border_y = height as i32 - EDGE_THRESHOLD + 3;

        let width_range = max_border_x - min_border_x;
        let height_range = max_border_y - min_border_y;

        let n_cols = width_range / w;
        let n_rows = height_range / w;

        let w_cell = ((width_range as f32 / n_cols as f32).ceil()) as i32;
        let h_cell = ((height_range as f32 / n_rows as f32).ceil()) as i32;

        let mut initial_keypoints = Vec::new();

        for i in 0..n_rows {
            let ini_y = min_border_y + i * h_cell;
            let max_y = ini_y + h_cell + 6;

            if ini_y >= max_border_y - 3 {
                continue;
            }

            for j in 0..n_cols {
                let ini_x = min_border_x + j * w_cell;
                let max_x = ini_x + w_cell + 6;

                if ini_x >= max_border_x - 6 {
                    continue;
                }

                let mut cell_keypoints = detect_keypoints_in_cell(
                    image,
                    ini_x.max(0) as u32,
                    ini_y.max(0) as u32,
                    max_x.min(width as i32) as u32,
                    max_y.min(height as i32) as u32,
                    ini_th_fast,
                    min_th_fast,
                );

                // Adjust coordinates back to full image space
                for kp in &mut cell_keypoints {
                    kp.pt.0 += ini_x as f32;
                    kp.pt.1 += ini_y as f32;
                    kp.octave = level as i32;
                }

                initial_keypoints.extend(cell_keypoints);
            }
        }

        // Redistribute keypoints using quadtree
        let mut distributed_keypoints = distribute_oct_tree(
            initial_keypoints,
            min_border_x,
            max_border_x,
            min_border_y,
            max_border_y,
            max_keypoints,
        );

        // Add scale information and compute orientation
        let scale_factor = pyramid.scale_factors[level];
        let scaled_patch_size = (PATCH_SIZE as f32 * scale_factor) as f32;

        let mut level_descriptors = Vec::with_capacity(distributed_keypoints.len());

        for mut kp in distributed_keypoints {
            kp.size = scaled_patch_size;
            kp.angle = compute_orientation(image, &mut kp);

            let descriptor = compute_orb_descriptor(image, &kp);
            level_descriptors.push(OrbDescriptor {
                keypoint: kp,
                descriptor,
            });
        }

        all_descriptors.push(level_descriptors);
    }

    all_descriptors
}

fn distribute_oct_tree(
    keypoints: Vec<KeyPoint>,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    desired_points: usize,
) -> Vec<KeyPoint> {
    if keypoints.is_empty() {
        return Vec::new();
    }

    // aspect ratio based node
    let width = max_x - min_x;
    let height = max_y - min_y;
    let n_ini_nodes = (width as f32 / height as f32).round() as i32;
    let h_x = width as f32 / n_ini_nodes as f32;

    let mut nodes = Vec::new();

    for i in 0..n_ini_nodes {
        let node = ExtractorNode::new(
            (min_x + (i as f32 * h_x) as i32, min_y),
            (min_x + ((i + 1) as f32 * h_x) as i32, min_y),
            (min_x + (i as f32 * h_x) as i32, max_y),
            (min_x + ((i + 1) as f32 * h_x) as i32, max_y),
        );
        nodes.push(node);
    }

    for kp in keypoints {
        let node_idx = ((kp.pt.0 as f32 - min_x as f32) / h_x) as usize;
        if node_idx < nodes.len() {
            nodes[node_idx].keypoints.push(kp);
        }
    }

    nodes.retain_mut(|node| {
        if node.keypoints.len() == 1 {
            node.no_more = true;
            true
        } else {
            !node.keypoints.is_empty()
        }
    });

    while nodes.len() < desired_points {
        let prev_size = nodes.len();
        let mut new_nodes = Vec::new();

        // Split nodes with most points first
        nodes.sort_by_key(|node| -(node.keypoints.len() as i32));

        for node in nodes {
            if node.no_more || node.keypoints.len() <= 1 {
                new_nodes.push(node);
                continue;
            }

            let (mut n1, mut n2, mut n3, mut n4) = node.divide();

            // Distribute points to children
            for kp in node.keypoints {
                let pt = kp.pt;
                if (pt.0 as i32) < n1.ur.0 {
                    if (pt.1 as i32) < n1.bl.1 {
                        n1.keypoints.push(kp);
                    } else {
                        n3.keypoints.push(kp);
                    }
                } else if (pt.1 as i32) < n1.bl.1 {
                    n2.keypoints.push(kp);
                } else {
                    n4.keypoints.push(kp);
                }
            }

            for mut n in [n1, n2, n3, n4] {
                if n.keypoints.len() == 1 {
                    n.no_more = true;
                }
                if !n.keypoints.is_empty() {
                    new_nodes.push(n);
                }
            }
        }

        nodes = new_nodes;

        if nodes.len() == prev_size {
            break;
        }
    }

    let mut result = Vec::with_capacity(desired_points);
    for node in nodes {
        if let Some(best_kp) = node
            .keypoints
            .iter()
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
        {
            result.push(best_kp.clone());
        }
    }

    result
}

fn detect_keypoints_in_cell(
    image: &GrayImage,
    ini_x: u32,
    ini_y: u32,
    max_x: u32,
    max_y: u32,
    ini_th_fast: u8,
    min_th_fast: u8,
) -> Vec<KeyPoint> {
    let sub_image = image.view(ini_x, ini_y, max_x - ini_x, max_y - ini_y);
    let cell_image = sub_image.to_image();
    let mut corners = corners_fast9(&cell_image, ini_th_fast);

    if corners.is_empty() {
        corners = corners_fast9(&cell_image, min_th_fast);
    }

    corners
        .into_iter()
        .map(|corner| KeyPoint {
            pt: (corner.x as f32, corner.y as f32),
            size: 7.0,
            angle: -1.0,
            response: corner.score as f32,
            octave: 0,
        })
        .collect()
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open("./test/fast_test.jpeg")?;
    let gray_img = img.to_luma8();
    let pyramid = ImagePyramid::new(&gray_img, 8, 1.2);

    // Detect keypoints and compute descriptors
    let all_descriptors = compute_descriptor_quad_tree(&pyramid, 20, 7, 200);

    let mut colored_img = img.to_rgb8();
    let mut total_keypoints = 0;

    let colors = [
        Rgb([255, 0, 0]),     // Red
        Rgb([0, 255, 0]),     // Green
        Rgb([0, 0, 255]),     // Blue
        Rgb([255, 255, 0]),   // Yellow
        Rgb([255, 0, 255]),   // Magenta
        Rgb([0, 255, 255]),   // Cyan
        Rgb([128, 128, 128]), // Gray
        Rgb([255, 128, 0]),   // Orange
    ];

    // Draw keypoints and print statistics
    println!("\nKeypoint Distribution across pyramid levels:");
    println!("Level | #Keypoints | Scale Factor");
    println!("------|------------|-------------");

    for (level, descriptors) in all_descriptors.iter().enumerate() {
        let color = colors[level % colors.len()];
        let scale_factor = pyramid.scale_factors[level];

        println!(
            "{:5} | {:10} | {:11.2}",
            level,
            descriptors.len(),
            scale_factor
        );
        total_keypoints += descriptors.len();

        for desc in descriptors {
            let kp = &desc.keypoint;
            let x = (kp.pt.0 * scale_factor) as i32 - EDGE_THRESHOLD;
            let y = (kp.pt.1 * scale_factor) as i32 - EDGE_THRESHOLD;

            draw_cross_mut(&mut colored_img, color, x, y);

            if true {
                let angle_rad = kp.angle * std::f32::consts::PI / 180.0;
                let line_length = (kp.size / 2.0) as i32;
                let end_x = x + (angle_rad.cos() * line_length as f32) as i32;
                let end_y = y + (angle_rad.sin() * line_length as f32) as i32;
                imageproc::drawing::draw_line_segment_mut(
                    &mut colored_img,
                    (x as f32, y as f32),
                    (end_x as f32, end_y as f32),
                    color,
                );
            }
        }
    }

    println!("------|------------|-------------");
    println!("Total keypoints: {}", total_keypoints);

    colored_img.save("./output/orb_features.png")?;
    println!("\nVisualization saved as '200_orb_features.png'");

    if false {
        use std::fs::File;
        use std::io::Write;
        let mut desc_file = File::create("./output/descriptors.txt")?;
        for (level, descriptors) in all_descriptors.iter().enumerate() {
            writeln!(desc_file, "Level {}:", level)?;
            for (idx, desc) in descriptors.iter().enumerate() {
                writeln!(
                    desc_file,
                    "Keypoint {}: x={:.1}, y={:.1}, angle={:.1}, size={:.1}",
                    idx,
                    desc.keypoint.pt.0,
                    desc.keypoint.pt.1,
                    desc.keypoint.angle,
                    desc.keypoint.size
                )?;
                writeln!(desc_file, "Descriptor: {:02X?}", desc.descriptor)?;
            }
        }
    }

    Ok(())
}
