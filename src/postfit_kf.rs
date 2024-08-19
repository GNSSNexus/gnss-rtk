use nyx::{
    linalg::{OMatrix, OVector, U3},
    od::prelude::{KfEstimate, KF},
    time::Epoch,
    State,
};

#[derive(Default, Copy, Clone, PartialEq)]
pub struct State3D {
    t: Epoch,
    x_km: f64,
    y_km: f64,
    z_km: f64,
}

impl std::fmt::Display for State3D {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{} x={:.5}km, y={:.5}km, z={:.5}km",
            self.t, self.x_km, self.y_km, self.z_km
        )
    }
}

impl std::fmt::LowerExp for State3D {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{} x={:.5}km, y={:.5}km, z={:.5}km",
            self.t, self.x_km, self.y_km, self.z_km
        )
    }
}

impl State for State3D {
    type Size = U3;
    type VecLength = U3;
    fn to_vector(&self) -> OVector<f64, U3> {
        OVector::<f64, U3>::new(self.x_km, self.y_km, self.z_km)
    }
    fn unset_stm(&mut self) {
        self.x_km = 0.0;
        self.y_km = 0.0;
        self.z_km = 0.0;
    }
    fn set(&mut self, epoch: Epoch, vector: &OVector<f64, U3>) {
        self.x_km = vector[0];
        self.y_km = vector[1];
        self.z_km = vector[2];
    }
    fn epoch(&self) -> Epoch {
        self.t
    }
    fn set_epoch(&mut self, epoch: Epoch) {
        self.t = epoch;
    }
}

pub struct PostFitKF {
    kf: KF<State3D, U3, U3>,
}

impl PostFitKF {
    /// Initialize new PostFitKF with initial target
    pub fn new(t: Epoch, x_km: f64, y_km: f64, z_km: f64) -> Self {
        let estimate = KfEstimate::from_diag(
            State3D {
                t,
                x_km,
                y_km,
                z_km,
            },
            OVector::<f64, U3>::new(1.0, 1.0, 1.0),
        );
        let noise = OMatrix::<f64, U3, U3>::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        Self {
            kf: KF::no_snc(estimate, noise),
        }
    }
}
