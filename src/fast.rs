use nalgebra as na;

pub struct KeyPoint {
    pub pt: na::Point2<f32>,
    pub response: f32,
}

pub fn fast(image: &na::DMatrix<u8>, threshold: u8, non_max_suppression: bool) -> Vec<KeyPoint> {
    let (rows, cols) = image.shape();
    let mut key_points = Vec::new();

    // Define the 16 pixels in a Bresenham circle of radius 3
    const CIRCLE: [(i32, i32); 16] = [
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
    ];

    for y in 3..rows - 3 {
        for x in 3..cols - 3 {
            let center = image[(y, x)];
            let mut brighter = 0;
            let mut darker = 0;

            for i in 0..16 {
                let pixel = image[(y + CIRCLE[i].1 as usize, x + CIRCLE[i].0 as usize)];
                if pixel >= center + threshold {
                    brighter += 1;
                    darker = 0;
                } else if pixel <= center - threshold {
                    darker += 1;
                    brighter = 0;
                } else {
                    brighter = 0;
                    darker = 0;
                }

                if brighter >= 9 || darker >= 9 {
                    key_points.push(KeyPoint {
                        pt: na::Point2::new(x as f32, y as f32),
                        response: (center as i16 - pixel as i16).abs() as f32,
                    });
                    break;
                }
            }
        }
    }

    if non_max_suppression {
        key_points.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        key_points.dedup_by(|a, b| {
            let dx = a.pt.x - b.pt.x;
            let dy = a.pt.y - b.pt.y;
            dx * dx + dy * dy < 10.0
        });
    }

    key_points
}

fn main() {
    let image = na::DMatrix::from_fn(100, 100, |i, j| ((i + j) % 256) as u8);

    let key_points = fast(&image, 20, true);
    println!("Detected {} key points", key_points.len());
}
