/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;


/**
 * Helper function computing the Gaussian probability directly taken from course material
 */
double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}


void ParticleFilter::init(double x, double y, double theta, double std[]) {

  // Number of particles
  num_particles = 200;
  double init_weight = 1.0 / num_particles;

  // Initialize random number generator and noise distributions
  default_random_engine gen;
  normal_distribution<double> noise_x(0.0, std[0]);
  normal_distribution<double> noise_y(0.0, std[1]);
  normal_distribution<double> noise_theta(0.0, std[2]);

  // Create particles
  particles.clear();
  for (int i = 0; i < num_particles; ++i) {
    Particle p {}; 
    p.id = i;
    p.weight = init_weight;
    // Initialize positions and heading
    p.x = x;
    p.y = y;
    p.theta = theta;
    // Add noise
    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
    // Append to vectors
    particles.push_back(p);
    weights.push_back(init_weight);
  }

  // Finish initializtion
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  // Initialize random number generator and noise distributions
  default_random_engine gen;
  normal_distribution<double> noise_x(0.0, std_pos[0]);
  normal_distribution<double> noise_y(0.0, std_pos[1]);
  normal_distribution<double> noise_theta(0.0, std_pos[2]);

  for (unsigned int i = 0; i < particles.size(); ++i) {
    Particle& p = particles[i];
    // Compute new positions and heading
    if (fabs(yaw_rate) < 1e-8) {
      p.x += velocity * delta_t;
      p.y += velocity * delta_t;
    } else {
      double theta_0 = p.theta;
      double theta_1 = p.theta + yaw_rate * delta_t;
      p.x += velocity / yaw_rate * (sin(theta_1) - sin(theta_0));
      p.y += velocity / yaw_rate * (cos(theta_0) - cos(theta_1));
      p.theta += yaw_rate * delta_t;
    }
    // Add noise
    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  // For each observation find closest prediction, i.e., landmarks in
  // sensor range, and store its vector position in observation id field
  for (unsigned int i = 0; i < observations.size(); ++i) {
    LandmarkObs& obs = observations[i];
    obs.id = -1;
    double minimum = __DBL_MAX__;
    for (unsigned int j = 0; j < predicted.size(); ++j) {
      const LandmarkObs& lm = predicted[j];
      double d = dist(obs.x, obs.y, lm.x, lm.y);
      if (d < minimum) {
        obs.id = j;
        minimum = d;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // Consult the following item for further understanding:
  // https://knowledge.udacity.com/questions/708237
  
  double norm_weight = 0.0;

  for (unsigned int i = 0; i < particles.size(); ++i) {
    Particle& p = particles[i];

    // Transform observations from particle to map coordinates
    vector<LandmarkObs> tobservations;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      const LandmarkObs& o = observations[j];
      LandmarkObs tobs {};
      tobs.id = -1;
      tobs.x = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
      tobs.y = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;
      tobservations.push_back(tobs);
    }

    // Filter landmarks within sensor range
    vector<LandmarkObs> predicted;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      const Map::single_landmark_s& lm = map_landmarks.landmark_list[j];
      if (dist(p.x, p.y, lm.x_f, lm.y_f) <= sensor_range) {
        LandmarkObs obs_lm {};
        obs_lm.id = lm.id_i;
        obs_lm.x = lm.x_f;
        obs_lm.y = lm.y_f;
        predicted.push_back(obs_lm);
      }
    }

    // Data association
    dataAssociation(predicted, tobservations);

    // Compute particle weight
    p.weight = 1.0;
    for (unsigned int j = 0; j < tobservations.size(); ++j) {
      const LandmarkObs& tobs = tobservations[j];
      if (tobs.id >= 0) {
        const LandmarkObs& lm = predicted[tobs.id];
        p.weight *= multiv_prob(
          std_landmark[0], std_landmark[1], tobs.x, tobs.y, lm.x, lm.y
        ); 
      } else {
        p.weight *= 0.0;
      }      
    }

    // Update overall normalization
    norm_weight += p.weight;

  }

  // Normaliztion
  for (unsigned int i = 0; i < particles.size(); ++i) {
    particles[i].weight /=  norm_weight;
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {

  // Initialize random number generator and sampling distribution
  default_random_engine gen;
  std::discrete_distribution<unsigned int> dist_w(weights.begin(), weights.end());

  // Create new particles
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; ++i) {
    new_particles.push_back(particles[dist_w(gen)]);
  }
  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}