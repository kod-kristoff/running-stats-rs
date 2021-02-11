#[derive(Clone, Copy, Debug)]
pub struct RunningMeanVar {
    n: usize,
    m1: f64,
    m2: f64,
}

impl RunningMeanVar {
    pub fn new() -> Self {
        Self { 
            n: 0,
            m1: 0.0,
            m2: 0.0,
        }
    }

    pub fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;

        let delta = x - self.m1;
        let delta_n = delta / self.n as f64;

        self.m1 += delta_n;
        self.m2 += delta * delta_n * n1;
    }

    pub fn num_data_points(&self) -> usize {
        self.n
    }

    pub fn mean(&self) -> f64 {
        self.m1
    }

    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            0.0
        }
    }

    pub fn std_deviation(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl std::ops::Add for RunningMeanVar {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let n = self.n + other.n;
        let m1 = (self.n as f64 * self.m1 + other.n as f64 * other.m1) / n as f64;

        let delta = other.m1 - self.m1;
        let m2 = self.m2 + other.m2 + delta * delta * (self.n * other.n) as f64 / n as f64;

        Self { n, m1, m2 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RunningStats {
    n: isize,
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self { 
            n: 0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    pub fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;

        let delta = x - self.m1;
        let delta_n = delta / self.n as f64;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;

        self.m1 += delta_n;
        self.m4 += term1 * delta_n2 * (self.n*self.n - 3*self.n + 3) as f64 + 6.0 * delta_n2 * self.m2 - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (self.n - 2) as f64 - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    pub fn num_data_points(&self) -> isize {
        self.n
    }

    pub fn mean(&self) -> f64 {
        self.m1
    }

    pub fn variance(&self) -> f64 {
        if self.n > 1 {
            self.m2 / (self.n - 1) as f64
        } else {
            0.0
        }
    }

    pub fn std_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn skewness(&self) -> f64 {
        if self.n > 2 {
            (self.n as f64).sqrt() * self.m3 / self.m2.powf(1.5)
        } else {
            0.0
        }
    }

    pub fn kurtosis(&self) -> f64 {
        if self.n < 2 {
            return 3.0;
        }
        self.n as f64 * self.m4 / (self.m2 * self.m2)
    }

    pub fn excess_kurtosis(&self) -> f64 {
        self.kurtosis() - 3.0
    }
}

impl std::ops::Add for RunningStats {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let n = self.n + other.n;
        let m1 = (self.m1 * self.n as f64 + other.m1 * other.n as f64) / n as f64;

        let delta = other.m1 - self.m1;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let m2 = self.m2 + other.m2 + delta2 * (self.n * other.n) as f64 / n as f64;
        let m3 = self.m3 + other.m3 + delta3 * (self.n * other.n * (self.n - other.n)) as f64 / (n * n) as f64;
        let m3 = m3 + 3.0 * delta * (self.n as f64 * other.m2 - other.n as f64 * self.m2) / n as f64;
        let m4 = self.m4 + other.m4 + delta4 * (self.n*other.n * (self.n*self.n - self.n*other.n + other.n*other.n)) as f64 / (n * n * n) as f64;
        let m4 = m4 + 6.0 * delta2 * ((self.n*self.n) as f64 * other.m2 + (other.n*other.n) as f64 * self.m2) / (n*n) as f64 + 4.0 * delta * (self.n as f64 * other.m3 - other.n as f64 * self.m3) / n as f64;
        
        Self { n, m1, m2, m3, m4 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod running_mean_var {
        use super::*;

        #[test]
        fn create_empty() {
            let running_mean_var = RunningMeanVar::new();
    
            assert_eq!(running_mean_var.num_data_points(), 0);
            assert_eq!(running_mean_var.mean(), 0.0);
            assert_eq!(running_mean_var.variance(), 0.0_f64);
            assert_eq!(running_mean_var.std_deviation(), 0.0_f64);
        }
        
        #[test]
        fn one_push() {
            let mut running_mean_var = RunningMeanVar::new();
    
            running_mean_var.push(2.0);
    
            assert_eq!(running_mean_var.num_data_points(), 1);
            assert_eq!(running_mean_var.mean(), 2.0);
            assert_eq!(running_mean_var.variance(), 0.0_f64);
            assert_eq!(running_mean_var.std_deviation(), 0.0_f64);
            // approx::assert_relative_eq()
        }
        
        #[test]
        fn two_pushes() {
            let mut running_mean_var = RunningMeanVar::new();
    
            running_mean_var.push(2.0);
            running_mean_var.push(4.0);
    
            assert_eq!(running_mean_var.num_data_points(), 2);
            approx::assert_relative_eq!(running_mean_var.mean(), 3.0);
            approx::assert_relative_eq!(running_mean_var.variance(), 2.0);
            approx::assert_relative_eq!(running_mean_var.std_deviation(), 2.0_f64.sqrt());
        }    

        mod add {
            use super::*;

            #[test]
            fn add_two() {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                let mut r1 = RunningMeanVar::new();
                let mut r2 = RunningMeanVar::new();

                for x in &data[0..2] {
                    r1.push(*x);
                }
                for x in &data[2..] {
                    r2.push(*x);
                }
                let mut r3 = RunningMeanVar::new();
                for x in &data {
                    r3.push(*x);
                }

                let r12 = r1 + r2;

                assert_eq!(
                    r12.num_data_points(),
                    r3.num_data_points()
                );

                approx::assert_relative_eq!(r12.mean(), r3.mean());
                approx::assert_relative_eq!(r12.variance(), r3.variance());

            }
        }
    }

    mod running_stats {
        use super::*;

        #[test]
        fn create_empty() {
            let running_stats = RunningStats::new();
    
            assert_eq!(running_stats.num_data_points(), 0);
            assert_eq!(running_stats.mean(), 0.0);
            assert_eq!(running_stats.variance(), 0.0);
            assert_eq!(running_stats.std_deviation(), 0.0);
            assert_eq!(running_stats.skewness(), 0.0);
            assert_eq!(running_stats.kurtosis(), 3.0);
        }
        
        #[test]
        fn one_push() {
            let mut running_stats = RunningStats::new();
    
            running_stats.push(2.0);
    
            assert_eq!(running_stats.num_data_points(), 1);
            assert_eq!(running_stats.mean(), 2.0);
            assert_eq!(running_stats.variance(), 0.0);
            assert_eq!(running_stats.std_deviation(), 0.0);
            assert_eq!(running_stats.skewness(), 0.0);
            assert_eq!(running_stats.kurtosis(), 3.0);
            // approx::assert_relative_eq()
        }
        
        #[test]
        fn two_pushes() {
            let mut running_stats = RunningStats::new();
    
            running_stats.push(2.0);
            running_stats.push(4.0);
    
            assert_eq!(running_stats.num_data_points(), 2);
            approx::assert_relative_eq!(running_stats.mean(), 3.0);
            approx::assert_relative_eq!(running_stats.variance(), 2.0);
            approx::assert_relative_eq!(running_stats.std_deviation(), 2.0_f64.sqrt());
            assert_eq!(running_stats.skewness(), 0.0);
            assert_eq!(running_stats.kurtosis(), 1.0);
        }

        #[test]
        fn many_pushes() {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0];

            let mut running_stats = RunningStats::new();
            for x in &data {
                running_stats.push(*x);
            }
            let n = data.len() as f64;
            let expected_mean = data.iter().sum::<f64>() / n;
            let expected_variance = data.iter().map(|x| (x - expected_mean).powf(2.0)).sum::<f64>() / (n - 1.0);
            let expected_std_dev = expected_variance.sqrt();
            let expected_m3 = data
                .iter()
                .map(|x| (x - expected_mean).powf(3.0))
                .sum::<f64>();
            let expected_m4 = data
                .iter()
                .map(|x| (x - expected_mean).powf(4.0))
                .sum::<f64>();

            approx::assert_relative_eq!(running_stats.mean(), expected_mean);
            approx::assert_relative_eq!(running_stats.variance(), expected_variance);
            approx::assert_relative_eq!(running_stats.std_deviation(), expected_std_dev);
            approx::assert_relative_eq!(running_stats.m3, expected_m3);
            approx::assert_relative_eq!(running_stats.m4, expected_m4);
        }

        mod add {
            use super::*;

            #[test]
            fn add_two() {
                let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0];

                let mut r1 = RunningStats::new();
                let mut r2 = RunningStats::new();

                for x in &data[0..3] {
                    r1.push(*x);
                }
                for x in &data[3..] {
                    r2.push(*x);
                }
                let mut r3 = RunningStats::new();
                for x in &data {
                    r3.push(*x);
                }

                let r12 = r1 + r2;

                assert_eq!(r1.num_data_points(), 3);
                assert_eq!(
                    r12.num_data_points(),
                    r3.num_data_points()
                );

                approx::assert_relative_eq!(r12.mean(), r3.mean());
                approx::assert_relative_eq!(r12.variance(), r3.variance());
                approx::assert_relative_eq!(r12.skewness(), r3.skewness());
                approx::assert_relative_eq!(r12.kurtosis(), r3.kurtosis());

            }
        }
    }
}
