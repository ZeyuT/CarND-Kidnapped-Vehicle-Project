/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;	//Random engine.

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Rename std for better readability
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	
	num_particles = 100;	// Number of particles
	
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// Rename std for better readability
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	for (int i = 0; i < num_particles; i++) {
		// for better readability
		double x_0 = particles[i].x;
		double y_0 = particles[i].y;
		double theta_0 = particles[i].theta;
		
		// Calculate the bew position and heading, without noise.
		double x_f, y_f, theta_f;
		if (fabs(yaw_rate) < 0.00001) {
			x_f = x_0 + velocity * delta_t * cos(theta_0);
			y_f = y_0 + velocity * delta_t * sin(theta_0);
		}
		else {
			x_f = x_0 + velocity / yaw_rate * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
			y_f = y_0 + velocity / yaw_rate * (cos(theta_0) - cos(theta_0 + yaw_rate * delta_t));
			theta_f = theta_0 + yaw_rate * delta_t;
		}

		// Add random Gaussian noise.
		// Create normal distributions for x, y and theta.
		normal_distribution<double> dist_x(x_f, std_x);
		normal_distribution<double> dist_y(y_f, std_y);
		normal_distribution<double> dist_theta(theta_f, std_theta);
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter  ::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
		LandmarkObs obs = observations[i];
		
		double min_dis = numeric_limits<double>::max();
		int temp_id = -1;
		for (unsigned int j = 0; j < predicted.size(); j++) {
			LandmarkObs pred = predicted[j];
			double dis = dist(pred.x, pred.y, obs.x, obs.y);
			if (dis < min_dis) {
				min_dis = dis;
				temp_id = pred.id;
			}
		}
		observations[i].id = temp_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//For better readability
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	for (int i = 0; i < num_particles; i++) {
		//For better readability
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
	
		//Get landmarks in the map.
		vector<LandmarkObs> landmark_inrange;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landm_x = map_landmarks.landmark_list[j].x_f;
			float landm_y = map_landmarks.landmark_list[j].y_f;
			int landm_id = map_landmarks.landmark_list[j].id_i;
			//Only consider landmarks within sensor ranges.
			if (fabs(landm_x - p_x) <= sensor_range && fabs(landm_y - p_y) <= sensor_range) {
				landmark_inrange.push_back(LandmarkObs{ landm_id, landm_x, landm_y });
			}
		}

		// Transformation
		vector<LandmarkObs> transformation;
		for (unsigned int k = 0; k < observations.size(); k++) {
			double t_x = cos(p_theta)*observations[k].x - sin(p_theta)*observations[k].y + p_x;
			double t_y = sin(p_theta)*observations[k].x + cos(p_theta)*observations[k].y + p_y;
			transformation.push_back(LandmarkObs{ observations[k].id, t_x, t_y });
		}

		// perform dataAssociation for the predictions and transformed observations on current particle
		dataAssociation(landmark_inrange, transformation);
		
		// initialize weight
		particles[i].weight = 1.0;

		//Calculate weights with multivariate Gaussian
		//Calculate weights of each observation.
		double mu_x, mu_y, x_obs, y_obs;
		for (unsigned int m = 0; m < transformation.size(); m++) {
			for (unsigned int n = 0; n < landmark_inrange.size(); n++) {
				if (transformation[m].id == landmark_inrange[n].id) {
					mu_x = landmark_inrange[n].x;
					mu_y = landmark_inrange[n].y;
				}
			}
			x_obs = transformation[m].x;
			y_obs = transformation[m].y;
			// Calculate normalization term
			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

			// Calculate exponent
			double exponent = pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

			// Calculate weight using normalization terms and exponent
			double weight = gauss_norm * exp(-exponent);

			//Calculate total observations weight
			particles[i].weight *= weight;
		}	
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<double> w;	// list of weights
	for (int i = 0; i < num_particles; i++) {
		w.push_back(particles[i].weight);
	}

	// Generate starting index.
	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);
	
	//Get distribution of beta
	double beta = 0.0;
	double mw = *max_element(w.begin(), w.end());
	uniform_real_distribution<double> betadist(0.0, 2 * mw);

	//Generate new particles.
	vector<Particle> p2;
	for (int j = 0; j < num_particles; j++) {
		beta += betadist(gen);
		while (beta > w[index]) {
			beta -= w[index];
			index = (index + 1) % num_particles;
		}
		p2.push_back(particles[index]);
	}
	particles = p2;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
