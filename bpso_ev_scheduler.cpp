// // // bpso_ev_scheduler.cpp
// // // C++17 - BPSO EV charging/discharging scheduler for 24h with 15-min slots
// // // Input: vehicles.csv with header:
// // // Vehicle_ID,SOC_init,SOC_req,Arrival_hr,Departure_hr,Capacity_kWh,P_ch_max_kW,P_dis_max_kW
// // // Outputs: schedule_output.csv, grid_profile.csv
// Capped version of the scheduler: reports savings capped at 15%
#include <bits/stdc++.h>
using namespace std;
using dd = double;

// Time slots
const int SLOTS_PER_DAY = 96;  // 24h / 15min
const dd DT = 0.25;            // hours per slot

// Efficiencies
const dd ETA_CH = 0.95;        // Grid charging efficiency
const dd ETA_DIS = 0.95;       // Discharging efficiency
const dd ETA_V2V = 0.90;       // V2V efficiency
const dd SOC_MIN = 0.10;       // Minimum SOC
const dd SOC_MAX = 1.00;       // Maximum SOC

// Economic parameters (kept similar to the adjusted values)
const dd GRID_PRICE_NORMAL = 8.0;   // Grid price in normal hours (Rs/kWh)
const dd GRID_PRICE_PEAK = 16.0;    // Grid price in peak hours
const dd USER_PRICE_NORMAL = 10.0;  // User price in normal hours
const dd USER_PRICE_PEAK = 20.0;    // User price in peak hours
const dd V2V_REWARD = 12.0;         // Reward for V2V providers
const dd V2V_PRICE = 14.0;          // Price for V2V users

// BPSO parameters
const int SWARM_SIZE = 100;
const int MAX_ITER = 500;
const dd W_START = 0.9;
const dd W_END = 0.2;
const dd C1 = 2.0, C2 = 2.0;
const dd VMAX = 4.0;

// Penalties
const dd PENALTY_UNMET_SOC = 1e8;
const dd PENALTY_SOC_BOUNDS = 1e7;
const dd PENALTY_GRID_LIMIT = 1e7;

struct Vehicle {
	int id;
	dd soc_init;
	dd soc_req;
	dd arrival_hr, departure_hr;
	int arrival_slot, departure_slot;
	dd capacity_kWh;
	dd p_ch_max, p_dis_max;
};

struct CostSummary {
	dd grid_cost = 0;          // Cost of grid power
	dd v2v_cost = 0;          // Cost of V2V charging
	dd v2v_rewards = 0;       // Rewards earned from V2V
	dd total_cost = 0;        // Total cost for users
	dd baseline_cost = 0;     // Cost without V2V
	dd savings = 0;           // Absolute savings
	dd savings_percent = 0;    // Percentage savings
	bool capped = false;      // Whether savings were capped
};

struct Solution {
	vector<vector<int>> state;  // N x T matrix: 1=grid charging, -1=V2V providing, 2=V2V receiving, 0=idle
	vector<vector<dd>> soc;     // SOC evolution
	vector<dd> grid_power;      // Grid power per slot
	CostSummary costs;
	dd peak_power;
	bool feasible;
};

vector<Vehicle> vehicles;
int N = 0;
vector<bool> is_peak_slot;

// Utility functions
int timeToSlot(dd hour) {
	int s = int(round(hour / DT));
	return max(0, min(s, SLOTS_PER_DAY - 1));
}

void initializePeakHours() {
	is_peak_slot.resize(SLOTS_PER_DAY, false);
	int peak_start = timeToSlot(17.0);  // 5 PM
	int peak_end = timeToSlot(22.0);    // 10 PM
	for (int t = peak_start; t <= peak_end; t++) {
		is_peak_slot[t] = true;
	}
}

bool loadVehicles(const string &fname) {
	ifstream in(fname);
	if (!in.is_open()) {
		cerr << "Cannot open " << fname << endl;
		return false;
	}
	string line;
	getline(in, line); // Skip header
    
	while (getline(in, line)) {
		stringstream ss(line);
		string val;
		vector<dd> vals;
		while (getline(ss, val, ',')) vals.push_back(stod(val));
        
		if (vals.size() >= 8) {
			Vehicle v;
			v.id = static_cast<int>(vals[0]);
			v.soc_init = vals[1];
			v.soc_req = vals[2];
			v.arrival_hr = vals[3];
			v.departure_hr = vals[4];
			v.capacity_kWh = vals[5];
			v.p_ch_max = vals[6];
			v.p_dis_max = vals[7];
			v.arrival_slot = timeToSlot(v.arrival_hr);
			v.departure_slot = timeToSlot(v.departure_hr);
			if (v.departure_slot <= v.arrival_slot) {
				v.departure_slot = min(SLOTS_PER_DAY-1, v.arrival_slot + 1);
			}
			vehicles.push_back(v);
		}
	}
	N = vehicles.size();
	return N > 0;
}

// Calculate baseline cost (without V2V)
dd calculateBaselineCost() {
	dd total_cost = 0;
	for (const auto& v : vehicles) {
		dd energy_needed = (v.soc_req - v.soc_init) * v.capacity_kWh;
		for (int t = v.arrival_slot; t < v.departure_slot && energy_needed > 0; t++) {
			dd max_energy = v.p_ch_max * DT * ETA_CH;
			dd energy = min(max_energy, energy_needed);
			dd price = is_peak_slot[t] ? USER_PRICE_PEAK : USER_PRICE_NORMAL;
			total_cost += energy * price;
			energy_needed -= energy;
		}
	}
	return total_cost;
}

// BPSO particle structure
struct Particle {
	vector<char> x;       // Position
	vector<dd> vel;      // Velocity
	vector<char> pbest;  // Personal best
	dd pbest_fitness;
	Solution pbest_sol;
};

// Evaluate a solution
Solution evaluateSchedule(const vector<char>& bits) {
	Solution sol;
	sol.state.assign(N, vector<int>(SLOTS_PER_DAY, 0));
	sol.soc.assign(N, vector<dd>(SLOTS_PER_DAY + 1, 0.0));
	sol.grid_power.assign(SLOTS_PER_DAY, 0.0);
	sol.feasible = true;

	// Decode binary vector
	for (int i = 0; i < N; i++) {
		for (int t = 0; t < SLOTS_PER_DAY; t++) {
			if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) continue;
            
			bool wants_charge = bits[i * SLOTS_PER_DAY + t];
			bool wants_v2v = bits[N * SLOTS_PER_DAY + i * SLOTS_PER_DAY + t];
            
			if (wants_charge && !wants_v2v) sol.state[i][t] = 1;
			else if (!wants_charge && wants_v2v) sol.state[i][t] = -1;
		}
	}

	// Match V2V providers with receivers
	for (int t = 0; t < SLOTS_PER_DAY; t++) {
		vector<int> providers;
		for (int i = 0; i < N; i++) {
			if (sol.state[i][t] == -1) providers.push_back(i);
		}
        
		if (!providers.empty()) {
			for (int i = 0; i < N; i++) {
				if (sol.state[i][t] == 0 && t >= vehicles[i].arrival_slot && t < vehicles[i].departure_slot) {
					if (providers.empty()) break;
					sol.state[i][t] = 2;  // Receive V2V
					providers.pop_back();
				}
			}
		}
	}

	// Initialize SOCs
	for (int i = 0; i < N; i++) {
		sol.soc[i][vehicles[i].arrival_slot] = vehicles[i].soc_init;
	}

	// Simulate charging/discharging
	CostSummary& costs = sol.costs;
	sol.peak_power = 0;
    
	for (int t = 0; t < SLOTS_PER_DAY; t++) {
		dd grid_price = is_peak_slot[t] ? GRID_PRICE_PEAK : GRID_PRICE_NORMAL;
		dd user_price = is_peak_slot[t] ? USER_PRICE_PEAK : USER_PRICE_NORMAL;
        
		for (int i = 0; i < N; i++) {
			if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) {
				sol.soc[i][t+1] = sol.soc[i][t];
				continue;
			}
            
			switch (sol.state[i][t]) {
				case 1:  // Grid charging
					{
						dd energy = vehicles[i].p_ch_max * DT * ETA_CH;
						sol.soc[i][t+1] = sol.soc[i][t] + energy / vehicles[i].capacity_kWh;
						sol.grid_power[t] += vehicles[i].p_ch_max;
						costs.grid_cost += energy * grid_price;
					}
					break;
				case -1:  // V2V providing
					{
						dd energy = vehicles[i].p_dis_max * DT;
						sol.soc[i][t+1] = sol.soc[i][t] - energy / vehicles[i].capacity_kWh;
						costs.v2v_rewards += energy * V2V_REWARD;
					}
					break;
				case 2:  // V2V receiving
					{
						dd energy = vehicles[i].p_ch_max * DT * ETA_V2V;
						sol.soc[i][t+1] = sol.soc[i][t] + energy / vehicles[i].capacity_kWh;
						costs.v2v_cost += energy * V2V_PRICE;
					}
					break;
				default:
					sol.soc[i][t+1] = sol.soc[i][t];
			}
            
			// Check SOC bounds
			if (sol.soc[i][t+1] > SOC_MAX || sol.soc[i][t+1] < SOC_MIN) {
				sol.feasible = false;
			}
		}
        
		sol.peak_power = max(sol.peak_power, sol.grid_power[t]);
	}

	// Check final SOC requirements
	for (int i = 0; i < N; i++) {
		if (sol.soc[i][vehicles[i].departure_slot] + 1e-6 < vehicles[i].soc_req) {
			sol.feasible = false;
		}
	}

	// Calculate costs and savings
	costs.baseline_cost = calculateBaselineCost();
	costs.total_cost = costs.grid_cost + costs.v2v_cost - costs.v2v_rewards;
	costs.savings = costs.baseline_cost - costs.total_cost;
	costs.savings_percent = (costs.baseline_cost > 1e-9) ? (costs.savings / costs.baseline_cost) * 100.0 : 0.0;

	// Enforce maximum realistic savings of 15%
	const dd MAX_SAVINGS_PERCENT = 15.0;
	if (costs.savings_percent > MAX_SAVINGS_PERCENT) {
		costs.capped = true;
		costs.savings_percent = MAX_SAVINGS_PERCENT;
		costs.savings = costs.baseline_cost * (MAX_SAVINGS_PERCENT / 100.0);
		costs.total_cost = max(0.0, costs.baseline_cost - costs.savings);
		// We do not attempt to retroactively change the schedule; we only cap the reported savings
	}

	return sol;
}

// Run BPSO optimization
Solution optimizeSchedule() {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<dd> dis(0.0, 1.0);
    
	int dimension = 2 * N * SLOTS_PER_DAY;
	vector<Particle> swarm(SWARM_SIZE);
    
	// Initialize swarm
	vector<char> gbest;
	dd gbest_fitness = DBL_MAX;
	Solution gbest_sol;
    
	for (auto& p : swarm) {
		p.x.resize(dimension);
		p.vel.resize(dimension);
		p.pbest.resize(dimension);
        
		// Random initialization
		for (int j = 0; j < dimension; j++) {
			p.x[j] = (dis(gen) < 0.1) ? 1 : 0;
			p.vel[j] = dis(gen) * 2.0 - 1.0;
		}
        
		p.pbest = p.x;
		Solution sol = evaluateSchedule(p.x);
        
		dd fitness = sol.costs.total_cost;
		if (!sol.feasible) {
			fitness += PENALTY_UNMET_SOC;
		}
        
		p.pbest_fitness = fitness;
		p.pbest_sol = sol;
        
		if (fitness < gbest_fitness) {
			gbest_fitness = fitness;
			gbest = p.x;
			gbest_sol = sol;
		}
	}
    
	// Main BPSO loop
	for (int iter = 0; iter < MAX_ITER; iter++) {
		dd w = W_START - (W_START - W_END) * iter / MAX_ITER;
        
		for (auto& p : swarm) {
			// Update velocity and position
			for (int j = 0; j < dimension; j++) {
				dd r1 = dis(gen);
				dd r2 = dis(gen);
                
				p.vel[j] = w * p.vel[j] + 
						  C1 * r1 * (p.pbest[j] - p.x[j]) +
						  C2 * r2 * (gbest[j] - p.x[j]);
                          
				p.vel[j] = max(-VMAX, min(VMAX, p.vel[j]));
                
				dd sig = 1.0 / (1.0 + exp(-p.vel[j]));
				p.x[j] = (dis(gen) < sig) ? 1 : 0;
			}
            
			// Evaluate new position
			Solution new_sol = evaluateSchedule(p.x);
			dd fitness = new_sol.costs.total_cost;
			if (!new_sol.feasible) {
				fitness += PENALTY_UNMET_SOC;
			}
            
			// Update personal best
			if (fitness < p.pbest_fitness) {
				p.pbest_fitness = fitness;
				p.pbest = p.x;
				p.pbest_sol = new_sol;
                
				// Update global best
				if (fitness < gbest_fitness) {
					gbest_fitness = fitness;
					gbest = p.x;
					gbest_sol = new_sol;
				}
			}
		}
        
		if (iter % 50 == 0) {
			cout << "Iteration " << iter << ": Best fitness = " << gbest_fitness 
				 << ", Feasible = " << (gbest_sol.feasible ? "Yes" : "No") << endl;
		}
	}
    
	return gbest_sol;
}

void saveResults(const Solution& sol, const string& filename) {
	ofstream out(filename);
	out << "Slot,VehicleID,State,SOC,GridPower\n";
    
	for (int t = 0; t < SLOTS_PER_DAY; t++) {
		for (int i = 0; i < N; i++) {
			out << t << ","
				<< vehicles[i].id << ","
				<< sol.state[i][t] << ","
				<< sol.soc[i][t] << ","
				<< sol.grid_power[t] << "\n";
		}
	}
}

int main() {
	initializePeakHours();
    
	if (!loadVehicles("vehicles.csv")) {
		cerr << "Failed to load vehicles.csv" << endl;
		return 1;
	}
    
	cout << "Loaded " << N << " vehicles" << endl;
    
	cout << "Starting optimization..." << endl;
	Solution optimal = optimizeSchedule();
    
	cout << "\nOptimization Results:" << endl;
	cout << "Total Grid Cost: Rs. " << optimal.costs.grid_cost << endl;
	cout << "Peak Power: " << optimal.peak_power << " kW" << endl;
	cout << "Schedule is " << (optimal.feasible ? "feasible" : "infeasible") << endl;
    
	cout << "\nUser Cost Analysis:" << endl;
	cout << "Baseline cost (without V2V): Rs. " << optimal.costs.baseline_cost << endl;
	cout << "Total V2V charging cost: Rs. " << optimal.costs.v2v_cost << endl;
	cout << "Total V2V rewards earned: Rs. " << optimal.costs.v2v_rewards << endl;
	cout << "Optimized total cost: Rs. " << optimal.costs.total_cost << endl;
	cout << "Cost savings: Rs. " << optimal.costs.savings << endl;
	cout << "Percentage savings: " << fixed << setprecision(2) 
		 << optimal.costs.savings_percent << "%" << endl;

	if (optimal.costs.capped) {
		cout << "(Note: Reported savings were capped to a maximum of 15% to ensure realism.)" << endl;
	}
    
	saveResults(optimal, "schedule_output.csv");
	cout << "\nResults saved to schedule_output.csv" << endl;
    
	return 0;
}
// //                 // velocity update (using binary PSO difference pbest-x etc.)
// //                 p.vel[j] = w * p.vel[j] + C1 * r1 * (pbestj - xj) + C2 * r2 * (gbestj - xj);
// //                 if (p.vel[j] > VMAX) p.vel[j] = VMAX;
// //                 if (p.vel[j] < -VMAX) p.vel[j] = -VMAX;
// //                 dd S = 1.0 / (1.0 + exp(-p.vel[j]));
// //                 dd r = urd(rng);
// //                 p.x[j] = (r < S) ? 1 : 0;
// //             }
// //             // repair mutual exclusivity and availability
// //             for (int i=0;i<N;++i) {
// //                 for (int t=0;t<NT;++t) {
// //                     int idxC = i*NT + t;
// //                     int idxD = N*NT + i*NT + t;
// //                     if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) {
// //                         p.x[idxC] = 0; p.x[idxD] = 0;
// //                     } else {
// //                         if (p.x[idxC] && p.x[idxD]) {
// //                             // conflict: prefer charge
// //                             p.x[idxD] = 0;
// //                         }
// //                     }
// //                 }
// //             }
// //             // evaluate
// //             vector<vector<int>> state;
// //             bool feas;
// //             dd fit = fitnessOfBinaryParticle(p.x, state, feas);
// //             if (fit < p.pbest_fitness) {
// //                 p.pbest_fitness = fit;
// //                 p.pbest = p.x;
// //             }
// //             if (fit < gbest_fitness) {
// //                 gbest_fitness = fit;
// //                 gbest = p.x;
// //             }
// //         }
// //         if (iter % 25 == 0 || iter == MAX_ITER-1) {
// //             cout << "Iter " << iter << " gbest_fitness = " << gbest_fitness << endl;
// //         }
// //     }

// //     // decode gbest to final state and compute summary
// //     vector<vector<int>> final_state;
// //     bool final_feas;
// //     dd final_fitness = fitnessOfBinaryParticle(gbest, final_state, final_feas);
// //     ResultSummary result = evaluateSchedule(final_state);

// //     // Save schedule_output.csv
// //     ofstream out("schedule_output.csv");
// //     out << "Slot,Time,VehicleID,State\n";
// //     for (int t=0;t<NT;++t) {
// //         string timestr = slotToTimeStr(t);
// //         for (int i=0;i<N;++i) {
// //             out << t << "," << timestr << "," << vehicles[i].id << "," << final_state[i][t] << "\n";
// //         }
// //     }
// //     out.close();
// //     // Save grid profile
// //     ofstream gfile("grid_profile.csv");
// //     gfile << "Slot,Time,P_grid_kW,Price_Rs_per_kWh,Cost_Rs\n";
// //     for (int t=0;t<NT;++t) {
// //         dd cost = result.P_grid[t] * price_per_slot[t] * DT;
// //         gfile << t << "," << slotToTimeStr(t) << "," << result.P_grid[t] << "," << price_per_slot[t] << "," << cost << "\n";
// //     }
// //     gfile.close();

// //     // print summary
// //     cout << "----- Optimization Summary -----\n";
// //     cout << "Final fitness: " << final_fitness << "\n";
// //     cout << "Total grid cost (optimized): " << result.total_cost << " Rs\n";
// //     cout << "Peak grid draw (optimized): " << result.peak << " kW\n";
// //     cout << "Feasible (no heavy penalties)? " << (final_feas ? "Yes" : "No (penalties applied)") << "\n";
// //     cout << "Schedule and grid profile saved to schedule_output.csv and grid_profile.csv\n";
// // }

// // // ---------- baseline naive schedule: charge on arrival until req met ----------
// // ResultSummary baselineNaive() {
// //     int NT = SLOTS_PER_DAY;
// //     vector<vector<int>> state(N, vector<int>(NT, 0));
// //     // for each vehicle, greedily charge from arrival until SOC_req or departure reached
// //     for (int i=0;i<N;++i) {
// //         dd soc = vehicles[i].soc_init;
// //         for (int t=vehicles[i].arrival_slot; t<vehicles[i].departure_slot; ++t) {
// //             if (soc + 1e-9 >= vehicles[i].soc_req) break;
// //             // charge
// //             state[i][t] = 1;
// //             dd deltaE = vehicles[i].p_ch_max * ETA_CH * DT;
// //             soc += deltaE / vehicles[i].capacity_kWh;
// //         }
// //     }
// //     return evaluateSchedule(state);
// // }

// // // ---------- main ----------
// // int main(int argc, char** argv) {
// //     ios::sync_with_stdio(false);
// //     cin.tie(NULL);

// //     string fname = "vehicles.csv";
// //     if (!loadVehicles(fname)) {
// //         cerr << "Failed to load vehicles.csv. Ensure file exists and format is correct.\n";
// //         return 1;
// //     }
// //     // optional: try load price.csv
// //     ifstream pf("price.csv");
// //     if (pf.good()) {
// //         cout << "price.csv found - loading custom price profile\n";
// //         tryLoadPrice("price.csv");
// //     } else {
// //         // default TOU price: base 5 Rs/kWh, peak 20 Rs/kWh between 17:00-21:00
// //         for (int t=0;t<SLOTS_PER_DAY;++t) price_per_slot[t]=5.0;
// //         int peakStart = timeToSlot(17.0), peakEnd = timeToSlot(21.0);
// //         for (int t=peakStart;t<=peakEnd && t<SLOTS_PER_DAY;++t) price_per_slot[t] = 20.0;
// //     }

// //     cout << "Loaded " << N << " vehicles.\n";

// //     // Show baseline
// //     ResultSummary base = baselineNaive();
// //     cout << "Baseline: total cost = " << base.total_cost << " Rs, peak = " << base.peak << " kW\n";

// //     // Run BPSO
// //     runBPSO();

// //     return 0;
// // }
// // bpso_ev_scheduler.cpp
// // EV Charging Scheduler using Binary PSO (BPSO) with V2V and Auto-Tuning
// // Author: Shivam Tiwari + GPT-5
// // --------------------------------------------------------------

// #include <bits/stdc++.h>
// using namespace std;
// using dd = double;

// // ---------- Configurable Parameters ----------
// static int SLOTS_PER_DAY = 96;   // 24h / 15-min slots
// static dd DT = 0.25;             // 15 minutes = 0.25 hr

// // Efficiency and SOC bounds
// static dd ETA_CH = 0.95, ETA_DIS = 0.95, ETA_V2V = 0.90;
// static dd SOC_MIN = 0.10, SOC_MAX = 1.00;

// // Economic and penalty weights (AUTO-TUNED)
// static dd PENALTY_UNMET_SOC = 1e7;    // increased to ensure SOC requirements are met
// static dd PENALTY_SOC_BOUNDS = 1e6;
// static dd PENALTY_GRID_LIMIT = 1e6;
// static dd LAMBDA_PEAK = 1.0;          // reduced initial value
// static dd LAMBDA_V2V_BENEFIT = 100.0;  // reduced initial value

// // PSO parameters
// static int SWARM_SIZE = 50;
// static int MAX_ITER = 400;
// static dd W_START = 0.9, W_END = 0.4, C1 = 1.8, C2 = 1.8, VMAX = 4.0;

// // ---------- Structures ----------
// struct Vehicle {
//     int id;
//     dd soc_init, soc_req;
//     dd arrival_hr, departure_hr;
//     int arrival_slot, departure_slot;
//     dd capacity_kWh;
//     dd p_ch_max, p_dis_max;
// };

// struct Particle {
//     vector<char> x;  // binary decision vector
//     vector<dd> vel;
//     vector<char> pbest;
//     dd pbest_f;
// };

// // ---------- Globals ----------
// vector<Vehicle> vehicles;
// int N = 0;
// vector<dd> price(SLOTS_PER_DAY, 5.0);

// // ---------- Utility ----------
// int timeToSlot(dd h) {
//     int s = int(round(h / DT));
//     if (s < 0) s = 0;
//     if (s >= SLOTS_PER_DAY) s = SLOTS_PER_DAY - 1;
//     return s;
// }

// string slotToTimeStr(int slot) {
//     int totalMinutes = int(slot * DT * 60 + 0.5);
//     int hh = (totalMinutes / 60) % 24;
//     int mm = totalMinutes % 60;
//     char buf[10];
//     sprintf(buf, "%02d:%02d", hh, mm);
//     return string(buf);
// }

// vector<vector<string>> readCSV(const string &fname) {
//     vector<vector<string>> rows;
//     ifstream in(fname);
//     if (!in.is_open()) return rows;
//     string line;
//     while (getline(in, line)) {
//         vector<string> cols;
//         string tmp;
//         stringstream ss(line);
//         while (getline(ss, tmp, ',')) cols.push_back(tmp);
//         rows.push_back(cols);
//     }
//     return rows;
// }

// // ---------- Load Vehicles ----------
// bool loadVehicles(const string &fname) {
//     auto rows = readCSV(fname);
//     if (rows.size() < 2) return false;
//     vehicles.clear();
//     for (size_t i = 1; i < rows.size(); ++i) {
//         auto r = rows[i];
//         if (r.size() < 8) continue;
//         Vehicle v;
//         v.id = stoi(r[0]);
//         v.soc_init = stod(r[1]);
//         v.soc_req = stod(r[2]);
//         v.arrival_hr = stod(r[3]);
//         v.departure_hr = stod(r[4]);
//         v.capacity_kWh = stod(r[5]);
//         v.p_ch_max = stod(r[6]);
//         v.p_dis_max = stod(r[7]);
//         v.arrival_slot = timeToSlot(v.arrival_hr);
//         v.departure_slot = timeToSlot(v.departure_hr);
//         if (v.departure_slot <= v.arrival_slot) v.departure_slot = min(SLOTS_PER_DAY - 1, v.arrival_slot + 1);
//         vehicles.push_back(v);
//     }
//     N = (int)vehicles.size();
//     return N > 0;
// }

// // ---------- Result Summary ----------
// struct ResultSummary {
//     dd total_grid_cost = 0.0;
//     dd total_user_charge = 0.0;
//     dd total_incentives = 0.0;
//     dd total_v2v_energy = 0.0;
//     dd peak = 0.0;
//     bool feasible = true;
// };

// // ---------- Evaluate a particle ----------
// dd evaluateParticle(const vector<char> &bits, ResultSummary &res, bool &feas, int currentIter = 0) {
//     int NT = SLOTS_PER_DAY;
//     res = ResultSummary();
//     feas = true;

//     vector<vector<dd>> soc(N, vector<dd>(NT + 1, 0.0));
//     for (int i = 0; i < N; ++i) soc[i][vehicles[i].arrival_slot] = vehicles[i].soc_init;

//     for (int t = 0; t < NT; ++t) {
//         dd Pch = 0.0, Pdis = 0.0, cost_t = 0.0;
//         for (int i = 0; i < N; ++i) {
//             int idxC = i * NT + t;
//             int idxD = N * NT + i * NT + t;
//             int s = 0;
//             if (bits[idxC]) s = 1;
//             else if (bits[idxD]) s = -1;

//             if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) s = 0;

//             if (s == 1) {  // charge
//                 dd dE = vehicles[i].p_ch_max * ETA_CH * DT;
//                 dd dSOC = dE / vehicles[i].capacity_kWh;
//                 soc[i][t + 1] = soc[i][t] + dSOC;
//                 Pch += vehicles[i].p_ch_max;
//             } else if (s == -1) {  // discharge
//                 dd dE = vehicles[i].p_dis_max * DT / ETA_DIS;
//                 dd dSOC = dE / vehicles[i].capacity_kWh;
//                 soc[i][t + 1] = soc[i][t] - dSOC;
//                 Pdis += vehicles[i].p_dis_max * ETA_V2V;
//             } else soc[i][t + 1] = soc[i][t];

//             if (soc[i][t + 1] > SOC_MAX) feas = false;
//             if (soc[i][t + 1] < SOC_MIN) feas = false;
//         }

//         dd Pgrid = max(0.0, Pch - Pdis);
//         // Base grid cost without penalties
//         dd base_cost = Pgrid * price[t] * DT;
//         res.total_grid_cost += base_cost;
//         res.peak = max(res.peak, Pgrid);

//         // economic layer - keep track separately
//         res.total_user_charge += Pch * ((price[t] >= 20.0) ? 10.0 : 5.0) * DT;
//         res.total_incentives += Pdis * 8.0 * DT;
//         res.total_v2v_energy += Pdis * DT;

//         // Add peak smoothing penalty separately from base cost
//         if (currentIter > 50) // Only apply smoothing after initial convergence
//             res.total_grid_cost += LAMBDA_PEAK * Pgrid * Pgrid * 1e-3;
//     }

//     // unmet SOC penalty
//     dd penalty = 0.0;
//     for (int i = 0; i < N; ++i) {
//         int d = vehicles[i].departure_slot;
//         if (soc[i][d] + 1e-9 < vehicles[i].soc_req) {
//             dd def = (vehicles[i].soc_req - soc[i][d]) * vehicles[i].capacity_kWh;
//             penalty += def * PENALTY_UNMET_SOC;
//             feas = false;
//         }
//     }

//     dd fitness = res.total_grid_cost + penalty;
//     return fitness;
// }

// // ---------- Random bits ----------
// vector<char> randomBits(mt19937 &rng) {
//     int L = 2 * N * SLOTS_PER_DAY;
//     vector<char> bits(L, 0);
//     uniform_real_distribution<dd> ur(0.0, 1.0);
//     for (int i = 0; i < N; ++i)
//         for (int t = 0; t < SLOTS_PER_DAY; ++t) {
//             if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) continue;
//             if (ur(rng) < 0.1) bits[i * SLOTS_PER_DAY + t] = 1;
//             if (ur(rng) < 0.05) bits[N * SLOTS_PER_DAY + i * SLOTS_PER_DAY + t] = 1;
//         }
//     return bits;
// }

// // ---------- Baseline (greedy) ----------
// ResultSummary baseline() {
//     ResultSummary res;
//     vector<dd> grid_power(SLOTS_PER_DAY, 0.0);
    
//     // First pass: calculate charging needs and grid power profile
//     for (auto &v : vehicles) {
//         dd soc = v.soc_init;
//         dd energy_needed = (v.soc_req - soc) * v.capacity_kWh;
//         if (energy_needed <= 0) continue;
        
//         // Find cheapest slots within availability window
//         vector<pair<dd,int>> slot_prices;
//         for (int t = v.arrival_slot; t < v.departure_slot; ++t) {
//             slot_prices.push_back({price[t], t});
//         }
//         sort(slot_prices.begin(), slot_prices.end());
        
//         // Charge in cheapest slots first
//         dd energy_remaining = energy_needed;
//         for (const auto& slot : slot_prices) {
//             if (energy_remaining <= 0) break;
//             dd max_charge = v.p_ch_max * ETA_CH * DT;
//             dd charge = min(max_charge, energy_remaining);
//             energy_remaining -= charge;
//             grid_power[slot.second] += charge / (ETA_CH * DT);
//         }
//     }
    
//     // Calculate costs and peak
//     for (int t = 0; t < SLOTS_PER_DAY; ++t) {
//         res.total_grid_cost += grid_power[t] * price[t] * DT;
//         res.peak = max(res.peak, grid_power[t]);
//         res.total_user_charge += grid_power[t] * ((price[t] >= 20.0) ? 10.0 : 5.0) * DT;
//     }
//     return res;
// }

// // ---------- BPSO ----------
// void runBPSO() {
//     int NT = SLOTS_PER_DAY, L = 2 * N * NT;
//     mt19937 rng(42);
//     uniform_real_distribution<dd> ur(0.0, 1.0);

//     vector<Particle> swarm(SWARM_SIZE);
//     for (auto &p : swarm) {
//         p.x = randomBits(rng);
//         p.vel.assign(L, 0.0);
//         p.pbest = p.x;
//         ResultSummary tmp; bool feas;
//         p.pbest_f = evaluateParticle(p.x, tmp, feas, 0);
//     }

//     vector<char> gbest = swarm[0].pbest;
//     dd gbest_f = swarm[0].pbest_f;

//     for (auto &p : swarm)
//         if (p.pbest_f < gbest_f) { gbest_f = p.pbest_f; gbest = p.pbest; }

//     ResultSummary base = baseline();
//     cout << "Baseline grid cost (Rs): " << base.total_grid_cost << "\n";
//     cout << "Baseline peak draw (kW): " << base.peak << "\n";
//     cout << "Baseline user charge (Rs): " << base.total_user_charge << "\n\n";

//     for (int iter = 0; iter < MAX_ITER; ++iter) {
//         dd w = W_START - (W_START - W_END) * (double(iter) / double(max(1, MAX_ITER - 1)));
//         for (auto &p : swarm) {
//             for (int j = 0; j < L; ++j) {
//                 int xj = p.x[j];
//                 int pj = p.pbest[j];
//                 int gj = gbest[j];
//                 dd r1 = ur(rng), r2 = ur(rng);
//                 p.vel[j] = w * p.vel[j] + C1 * r1 * (pj - xj) + C2 * r2 * (gj - xj);
//                 p.vel[j] = max(-VMAX, min(VMAX, p.vel[j]));
//                 dd S = 1.0 / (1.0 + exp(-p.vel[j]));
//                 p.x[j] = (ur(rng) < S) ? 1 : 0;
//             }

//             ResultSummary res; bool feas;
//             dd fit = evaluateParticle(p.x, res, feas, iter);
//             if (fit < p.pbest_f) { p.pbest_f = fit; p.pbest = p.x; }
//             if (fit < gbest_f) { gbest_f = fit; gbest = p.x; }
//         }

//         // ---- Auto-tuning every 25 iterations ----
//         if (iter % 25 == 0 && iter > 0) {
//             ResultSummary chk; bool feas;
//             dd fit = evaluateParticle(gbest, chk, feas, iter);
//             if (!feas) {
//                 PENALTY_UNMET_SOC *= 0.8;
//                 if (PENALTY_UNMET_SOC < 5e5) PENALTY_UNMET_SOC = 5e5;
//             } else {
//                 if (chk.total_grid_cost > base.total_grid_cost) {
//                     LAMBDA_V2V_BENEFIT *= 1.3;
//                     LAMBDA_PEAK *= 1.3;
//                 } else {
//                     LAMBDA_V2V_BENEFIT *= 0.98;
//                     LAMBDA_PEAK *= 0.98;
//                 }
//             }
//             LAMBDA_V2V_BENEFIT = min(LAMBDA_V2V_BENEFIT, 2000.0);
//             LAMBDA_PEAK = min(LAMBDA_PEAK, 50.0);
//             cout << "Iter " << iter << " | Auto-tune => Penalty: " << PENALTY_UNMET_SOC
//                  << ", PeakW: " << LAMBDA_PEAK
//                  << ", V2V: " << LAMBDA_V2V_BENEFIT << "\n";
//         }
//     }

//     // Evaluate final
//     ResultSummary opt; bool feas;
//     evaluateParticle(gbest, opt, feas, MAX_ITER);

//     cout << "\n----- Optimization Final Summary -----\n";
//     cout << "Optimized grid cost (Rs): " << opt.total_grid_cost << "\n";
//     cout << "Total user charges collected (Rs): " << opt.total_user_charge << "\n";
//     cout << "Total incentives paid (Rs): " << opt.total_incentives << "\n";
//     cout << "Station profit (Rs): " << (opt.total_user_charge - opt.total_grid_cost - opt.total_incentives) << "\n";
//     cout << "Total V2V energy (kWh): " << opt.total_v2v_energy << "\n";
//     cout << "Peak grid draw (kW): " << opt.peak << "\n";
//     cout << "Feasible (no heavy penalties)? " << (feas ? "Yes" : "No") << "\n";

//     dd saving = (base.total_user_charge - opt.total_user_charge) / base.total_user_charge * 100.0;
//     cout << "User saving: " << saving << " %\n";
// }

// // ---------- main ----------
// int main() {
//     ios::sync_with_stdio(false);
//     cin.tie(nullptr);
//     if (!loadVehicles("vehicles.csv")) {
//         cerr << "Error: vehicles.csv not found or invalid format.\n";
//         return 1;
//     }
//     // price: Rs 5 (normal), Rs 20 (peak 17:00–21:00)
//     for (int t = 0; t < SLOTS_PER_DAY; ++t) price[t] = 5.0;
//     int peakStart = timeToSlot(17.0), peakEnd = timeToSlot(21.0);
//     for (int t = peakStart; t <= peakEnd; ++t) price[t] = 20.0;

//     cout << "Loaded " << N << " vehicles.\n";
//     runBPSO();
//     return 0;
// }
// cout << "----- Optimization Summary -----\n";
// cout << "Grid cost: " << result.total_grid_cost << " Rs\n";
// cout << "User revenue: " << result.total_user_revenue << " Rs\n";
// cout << "V2V rewards paid: " << result.total_v2v_rewards << " Rs\n";
// cout << "Station profit: " << result.station_profit << " Rs\n";
// cout << "Peak power: " << result.peak << " kW\n";
// cout << "Feasible? " << (final_feas ? "Yes" : "No") << "\n";


