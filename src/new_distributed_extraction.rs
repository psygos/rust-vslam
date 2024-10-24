use crate::pyramid::*;
use image::Rgb;
use image::{GenericImageView, GrayImage};
use imageproc::corners::{corners_fast9, Corner};
use imageproc::drawing::draw_cross_mut;

const EDGE_THRESHOLD: i32 = 19;

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

pub fn compute_keypoints_oct_tree(
    pyramid: &ImagePyramid,
    ini_th_fast: u8,
    min_th_fast: u8,
    max_keypoints: usize,
) -> Vec<Vec<KeyPoint>> {
    let mut all_keypoints = Vec::new();
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
        let distributed_keypoints = distribute_oct_tree(
            initial_keypoints,
            min_border_x,
            max_border_x,
            min_border_y,
            max_border_y,
            max_keypoints,
        );

        all_keypoints.push(distributed_keypoints);
    }

    all_keypoints
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

    // Initialize nodes based on aspect ratio
    let width = max_x - min_x;
    let height = max_y - min_y;
    let n_ini_nodes = (width as f32 / height as f32).round() as i32;
    let h_x = width as f32 / n_ini_nodes as f32;

    let mut nodes = Vec::new();

    // Create initial horizontal strips
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

            // Add non-empty nodes
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
    let all_keypoints = compute_keypoints_oct_tree(&pyramid, 20, 7, 1000);

    let mut colored_img = img.to_rgb8();

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

    for (level, keypoints) in all_keypoints.iter().enumerate() {
        let color = colors[level % colors.len()];
        let scale_factor = pyramid.scale_factors[level];
        for kp in keypoints {
            let x = (kp.pt.0 * scale_factor) as i32 - EDGE_THRESHOLD;
            let y = (kp.pt.1 * scale_factor) as i32 - EDGE_THRESHOLD;
            draw_cross_mut(&mut colored_img, color, x, y);
        }
    }

    colored_img.save("./output/new_keypoints_visualization.png")?;
    println!("Keypoint visualization saved as 'keypoints_visualization.png'");

    Ok(())
}
