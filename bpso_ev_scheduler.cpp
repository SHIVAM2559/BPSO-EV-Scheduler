// bpso_ev_scheduler.cpp
// C++17 - BPSO EV charging/discharging scheduler for 24h with 15-min slots
// Input: vehicles.csv with header:
// Vehicle_ID,SOC_init,SOC_req,Arrival_hr,Departure_hr,Capacity_kWh,P_ch_max_kW,P_dis_max_kW
// Outputs: schedule_output.csv, grid_profile.csv
//
// Compile:
// g++ -std=c++17 -O2 -pthread bpso_ev_scheduler.cpp -o scheduler
// Run:
// ./scheduler

#include <bits/stdc++.h>
using namespace std;
using dd = double;

// ---------- Configurable parameters ----------
const int SLOTS_PER_DAY = 96;       // 24h / 15min
const dd DT = 0.25;                 // hours per slot
const dd ETA_CH = 0.95;
const dd ETA_DIS = 0.95;
const dd ETA_V2V = 0.90;
const dd SOC_MIN = 0.10;
const dd SOC_MAX = 1.00;
const dd PENALTY_UNMET_SOC = 1e7;    // heavy penalty
const dd PENALTY_SOC_BOUNDS = 1e6;
const dd PENALTY_GRID_LIMIT = 1e6;
const dd LAMBDA_PEAK = 10.0;         // peak smoothing weight (on squared grid power)
const int SWARM_SIZE = 50;
const int MAX_ITER = 400;
const dd W_INERTIA_START = 0.9;
const dd W_INERTIA_END = 0.4;
const dd C1 = 1.8, C2 = 1.8;
const dd VMAX = 4.0; // for velocity limiting

// ---------- Data structures ----------
struct Vehicle {
    int id;
    dd soc_init;   // fraction 0..1
    dd soc_req;    // fraction 0..1
    dd arrival_hr, departure_hr;
    int arrival_slot, departure_slot;
    dd capacity_kWh;
    dd p_ch_max, p_dis_max; // kW
};

struct Particle {
    vector<char> x;      // binary vector (0/1) for 2*N*T bits (C then D)
    vector<dd> vel;      // velocities
    vector<char> pbest;  // personal best solution
    dd pbest_fitness;
};

// ---------- Utility functions ----------
int timeToSlot(dd hour) {
    // convert hour (0-24) to slot index 0..95
    int s = int(round(hour / DT));
    if (s < 0) s = 0;
    if (s > SLOTS_PER_DAY) s = SLOTS_PER_DAY;
    if (s == SLOTS_PER_DAY) s = SLOTS_PER_DAY - 1;
    return s;
}

string slotToTimeStr(int slot) {
    int totalMinutes = int(slot * DT * 60 + 0.5);
    int hh = (totalMinutes / 60) % 24;
    int mm = totalMinutes % 60;
    char buf[10];
    sprintf(buf, "%02d:%02d", hh, mm);
    return string(buf);
}

// read CSV
vector<vector<string>> readCSV(const string &fname) {
    vector<vector<string>> rows;
    ifstream in(fname);
    if (!in.is_open()) {
        cerr << "Cannot open file " << fname << endl;
        return rows;
    }
    string line;
    while (getline(in, line)) {
        // simple CSV split by comma (assume no commas inside fields)
        vector<string> cols;
        string cur;
        stringstream ss(line);
        while (getline(ss, cur, ',')) cols.push_back(cur);
        // trim spaces
        for (auto &c : cols) {
            // trim leading/trailing
            size_t start = c.find_first_not_of(" \t\r\n");
            size_t end = c.find_last_not_of(" \t\r\n");
            if (start==string::npos) c = "";
            else c = c.substr(start, end - start + 1);
        }
        rows.push_back(cols);
    }
    in.close();
    return rows;
}

// ---------- Global problem data ----------
vector<Vehicle> vehicles;
int N = 0;
vector<dd> price_per_slot(SLOTS_PER_DAY, 5.0); // default
vector<dd> grid_limit_slot(SLOTS_PER_DAY, 1e9); // large by default (no limit)

// ---------- Helper: load vehicles.csv ----------
bool loadVehicles(const string &fname) {
    auto rows = readCSV(fname);
    if (rows.empty()) return false;
    // expect header
    // find header row and start from next
    int start = 1;
    if (rows.size() < 2) return false;
    vehicles.clear();
    for (size_t i = start; i < rows.size(); ++i) {
        auto &r = rows[i];
        if (r.size() < 8) continue;
        Vehicle v;
        v.id = stoi(r[0]);
        v.soc_init = stod(r[1]);
        v.soc_req = stod(r[2]);
        v.arrival_hr = stod(r[3]);
        v.departure_hr = stod(r[4]);
        v.capacity_kWh = stod(r[5]);
        v.p_ch_max = stod(r[6]);
        v.p_dis_max = stod(r[7]);
        v.arrival_slot = timeToSlot(v.arrival_hr);
        v.departure_slot = timeToSlot(v.departure_hr);
        if (v.departure_slot <= v.arrival_slot) {
            // ensure at least one slot window; if departure same as arrival, set departure = arrival+1
            v.departure_slot = min(SLOTS_PER_DAY-1, v.arrival_slot + 1);
        }
        vehicles.push_back(v);
    }
    N = (int)vehicles.size();
    return N > 0;
}

// optional: try to read price.csv (slot,price) if present
void tryLoadPrice(const string &fname) {
    auto rows = readCSV(fname);
    if (rows.empty()) return;
    // assume header, then rows with slotIndex or hh:mm,price
    for (size_t i = 1; i < rows.size(); ++i) {
        auto &r = rows[i];
        if (r.size() < 2) continue;
        // if first column is slot index numeric
        try {
            int slot = stoi(r[0]);
            dd p = stod(r[1]);
            if (slot >= 0 && slot < SLOTS_PER_DAY) price_per_slot[slot] = p;
        } catch (...) {
            // maybe hh:mm
            string t = r[0];
            int hh=0, mm=0;
            if (sscanf(t.c_str(), "%d:%d", &hh, &mm) == 2) {
                int slot = int((hh*60 + mm) / (DT*60) + 0.5);
                if (slot >= 0 && slot < SLOTS_PER_DAY) price_per_slot[slot] = stod(r[1]);
            }
        }
    }
}

// compute baseline cost (naive: charge on arrival until SOC_req met)
struct ResultSummary {
    vector<vector<int>> state; // N x T {1 charge, -1 discharge, 0 idle}
    vector<vector<dd>> soc;    // N x (T+1) soc evolution
    vector<dd> P_grid;
    dd total_cost;
    dd peak;
    bool feasible;
};

ResultSummary evaluateSchedule(const vector<vector<int>> &state) {
    // state: N x T
    ResultSummary res;
    res.state = state;
    res.soc.assign(N, vector<dd>(SLOTS_PER_DAY+1, 0.0));
    res.P_grid.assign(SLOTS_PER_DAY, 0.0);
    res.total_cost = 0.0;
    res.peak = 0.0;
    res.feasible = true;

    // init SOC at arrival slot (we will fill soc[t] and soc[t+1])
    for (int i=0;i<N;++i) {
        int a = vehicles[i].arrival_slot;
        // before arrival, SOC unknown/irrelevant; set at arrival slot
        for (int t=0;t<=SLOTS_PER_DAY;++t) res.soc[i][t]=0.0;
        res.soc[i][vehicles[i].arrival_slot] = vehicles[i].soc_init;
    }

    for (int t=0;t<SLOTS_PER_DAY;++t) {
        dd P_charge_total = 0.0, P_dis_total = 0.0;
        // for each vehicle update SOC and accumulate charge/discharge
        for (int i=0;i<N;++i) {
            // if before arrival, keep SOC as initial (we set at arrival)
            if (t < vehicles[i].arrival_slot) {
                res.soc[i][t+1] = res.soc[i][t];
                continue;
            }
            if (t >= vehicles[i].departure_slot) {
                // after departure, SOC remains same (vehicle left)
                res.soc[i][t+1] = res.soc[i][t];
                continue;
            }
            int s = state[i][t];
            if (s == 1) {
                // charge
                dd deltaE = vehicles[i].p_ch_max * ETA_CH * DT; // kWh added
                dd deltaSOC = deltaE / vehicles[i].capacity_kWh;
                res.soc[i][t+1] = res.soc[i][t] + deltaSOC;
                P_charge_total += vehicles[i].p_ch_max;
            } else if (s == -1) {
                // discharge
                dd deltaE = vehicles[i].p_dis_max * DT / ETA_DIS; // kWh taken from battery (account efficiency)
                dd deltaSOC = deltaE / vehicles[i].capacity_kWh;
                res.soc[i][t+1] = res.soc[i][t] - deltaSOC;
                // discharging supply (after v2v efficiency)
                P_dis_total += vehicles[i].p_dis_max * ETA_V2V;
            } else {
                res.soc[i][t+1] = res.soc[i][t];
            }
        }
        // compute grid draw
        dd Pgrid = P_charge_total - P_dis_total;
        if (Pgrid < 0) Pgrid = 0.0; // clamp - no exporting
        res.P_grid[t] = Pgrid;
        // cost
        res.total_cost += Pgrid * price_per_slot[t] * DT;
        if (Pgrid > res.peak) res.peak = Pgrid;

        // check SOC bounds
        for (int i=0;i<N;++i) {
            if (res.soc[i][t+1] > SOC_MAX + 1e-9) {
                res.feasible = false;
            }
            if (res.soc[i][t+1] < SOC_MIN - 1e-9) {
                res.feasible = false;
            }
        }
    }

    // check departure SOC requirements
    for (int i=0;i<N;++i) {
        int dslot = vehicles[i].departure_slot;
        if (res.soc[i][dslot] + 1e-9 < vehicles[i].soc_req) {
            res.feasible = false;
        }
    }
    return res;
}

// ---------- Fitness function for PSO: lower is better ----------
dd fitnessOfBinaryParticle(const vector<char> &bits, // length 2*N*T : first N*T are C, next N*T are D
                           vector<vector<int>> &out_state, // will fill N x T with -1/0/1
                           bool &feasible_out) {
    // decode
    int NT = SLOTS_PER_DAY;
    out_state.assign(N, vector<int>(NT, 0));
    // enforce availability & mutual exclusivity repair
    for (int i=0;i<N;++i) {
        for (int t=0;t<NT;++t) {
            int idxC = i*NT + t;
            int idxD = N*NT + i*NT + t;
            char c = bits[idxC];
            char d = bits[idxD];
            // if outside availability -> force zero
            if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) {
                c = 0; d = 0;
            }
            if (c && d) {
                // conflict: randomly prefer charging (or prefer whichever cheaper?) here choose charge by default
                d = 0;
            }
            out_state[i][t] = (c ? 1 : (d ? -1 : 0));
        }
    }

    // simulate
    dd total_cost = 0.0;
    dd peak = 0.0;
    dd penalty = 0.0;
    vector<vector<dd>> soc(N, vector<dd>(NT+1, 0.0));
    for (int i=0;i<N;++i) soc[i][vehicles[i].arrival_slot] = vehicles[i].soc_init;

    for (int t=0;t<NT;++t) {
        dd P_charge_total = 0.0, P_dis_total = 0.0;
        for (int i=0;i<N;++i) {
            if (t < vehicles[i].arrival_slot) {
                soc[i][t+1] = soc[i][t];
                continue;
            }
            if (t >= vehicles[i].departure_slot) {
                soc[i][t+1] = soc[i][t];
                continue;
            }
            int s = out_state[i][t];
            if (s == 1) {
                dd deltaE = vehicles[i].p_ch_max * ETA_CH * DT;
                dd deltaSOC = deltaE / vehicles[i].capacity_kWh;
                soc[i][t+1] = soc[i][t] + deltaSOC;
                P_charge_total += vehicles[i].p_ch_max;
            } else if (s == -1) {
                dd deltaE = vehicles[i].p_dis_max * DT / ETA_DIS;
                dd deltaSOC = deltaE / vehicles[i].capacity_kWh;
                soc[i][t+1] = soc[i][t] - deltaSOC;
                P_dis_total += vehicles[i].p_dis_max * ETA_V2V;
            } else soc[i][t+1] = soc[i][t];
            // SOC bounds penalty
            if (soc[i][t+1] > SOC_MAX) penalty += PENALTY_SOC_BOUNDS * (soc[i][t+1] - SOC_MAX);
            if (soc[i][t+1] < SOC_MIN) penalty += PENALTY_SOC_BOUNDS * (SOC_MIN - soc[i][t+1]);
        }
        dd Pgrid = P_charge_total - P_dis_total;
        if (Pgrid < 0) Pgrid = 0;
        if (Pgrid > grid_limit_slot[t]) penalty += PENALTY_GRID_LIMIT * (Pgrid - grid_limit_slot[t]);
        total_cost += Pgrid * price_per_slot[t] * DT;
        peak = max(peak, Pgrid);
        // small smoothing penalty to discourage spikes (squared)
        penalty += LAMBDA_PEAK * Pgrid * Pgrid * 1e-3;
    }

    // departure SOC penalties
    dd unmet_sum = 0.0;
    for (int i=0;i<N;++i) {
        int dslot = vehicles[i].departure_slot;
        dd soc_at_depart = soc[i][dslot];
        if (soc_at_depart + 1e-9 < vehicles[i].soc_req) {
            dd deficit_kWh = (vehicles[i].soc_req - soc_at_depart) * vehicles[i].capacity_kWh;
            unmet_sum += deficit_kWh;
        }
    }
    if (unmet_sum > 0) penalty += PENALTY_UNMET_SOC * unmet_sum;

    feasible_out = (penalty < 1e-3);
    dd fitness = total_cost + penalty;
    return fitness;
}

// ---------- Initialize particle binary vector randomly with availability respected ----------
vector<char> randomInitialBits(mt19937 &rng) {
    int NT = SLOTS_PER_DAY;
    int L = 2 * N * NT;
    vector<char> bits(L, 0);
    uniform_real_distribution<dd> ud(0.0, 1.0);
    for (int i=0;i<N;++i) {
        for (int t=0;t<NT;++t) {
            if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) {
                bits[i*NT + t] = 0;
                bits[N*NT + i*NT + t] = 0;
            } else {
                // randomly set charge/discharge/idle with low probability to avoid infeasible huge loads
                double r = ud(rng);
                if (r < 0.10) bits[i*NT + t] = 1;        // 10% chance charge
                else bits[i*NT + t] = 0;
                r = ud(rng);
                if (r < 0.05) bits[N*NT + i*NT + t] = 1; // 5% chance discharge
                else bits[N*NT + i*NT + t] = 0;
                if (bits[i*NT + t] && bits[N*NT + i*NT + t]) {
                    // conflict -> prefer charge
                    bits[N*NT + i*NT + t] = 0;
                }
            }
        }
    }
    return bits;
}

// ---------- BPSO main ----------
void runBPSO() {
    if (N == 0) {
        cerr << "No vehicles loaded.\n";
        return;
    }
    int NT = SLOTS_PER_DAY;
    int L = 2 * N * NT;
    mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<dd> urd(0.0, 1.0);

    // Initialize swarm
    vector<Particle> swarm(SWARM_SIZE);
    for (int s=0; s<SWARM_SIZE; ++s) {
        swarm[s].x = randomInitialBits(rng);
        swarm[s].vel.assign(L, 0.0);
        swarm[s].pbest = swarm[s].x;
        // evaluate pbest fitness
        vector<vector<int>> tmpstate;
        bool feas;
        swarm[s].pbest_fitness = fitnessOfBinaryParticle(swarm[s].pbest, tmpstate, feas);
        // store pbest even if infeasible - PSO will try to improve
    }

    // global best
    vector<char> gbest = swarm[0].pbest;
    dd gbest_fitness = swarm[0].pbest_fitness;
    for (int s=1; s<SWARM_SIZE; ++s) {
        if (swarm[s].pbest_fitness < gbest_fitness) {
            gbest_fitness = swarm[s].pbest_fitness;
            gbest = swarm[s].pbest;
        }
    }

    cout << "Initial best fitness: " << gbest_fitness << endl;

    // PSO loop
    for (int iter=0; iter<MAX_ITER; ++iter) {
        // inertia linear decrease
        dd w = W_INERTIA_START - (W_INERTIA_START - W_INERTIA_END) * (double(iter)/double(max(1,MAX_ITER-1)));
        for (int s=0; s<SWARM_SIZE; ++s) {
            Particle &p = swarm[s];
            // update velocities and positions (binary PSO)
            for (int j=0;j<L;++j) {
                int xj = (p.x[j] ? 1 : 0);
                int pbestj = (p.pbest[j] ? 1 : 0);
                int gbestj = (gbest[j] ? 1 : 0);
                dd r1 = urd(rng);
                dd r2 = urd(rng);
                // velocity update (using binary PSO difference pbest-x etc.)
                p.vel[j] = w * p.vel[j] + C1 * r1 * (pbestj - xj) + C2 * r2 * (gbestj - xj);
                if (p.vel[j] > VMAX) p.vel[j] = VMAX;
                if (p.vel[j] < -VMAX) p.vel[j] = -VMAX;
                dd S = 1.0 / (1.0 + exp(-p.vel[j]));
                dd r = urd(rng);
                p.x[j] = (r < S) ? 1 : 0;
            }
            // repair mutual exclusivity and availability
            for (int i=0;i<N;++i) {
                for (int t=0;t<NT;++t) {
                    int idxC = i*NT + t;
                    int idxD = N*NT + i*NT + t;
                    if (t < vehicles[i].arrival_slot || t >= vehicles[i].departure_slot) {
                        p.x[idxC] = 0; p.x[idxD] = 0;
                    } else {
                        if (p.x[idxC] && p.x[idxD]) {
                            // conflict: prefer charge
                            p.x[idxD] = 0;
                        }
                    }
                }
            }
            // evaluate
            vector<vector<int>> state;
            bool feas;
            dd fit = fitnessOfBinaryParticle(p.x, state, feas);
            if (fit < p.pbest_fitness) {
                p.pbest_fitness = fit;
                p.pbest = p.x;
            }
            if (fit < gbest_fitness) {
                gbest_fitness = fit;
                gbest = p.x;
            }
        }
        if (iter % 25 == 0 || iter == MAX_ITER-1) {
            cout << "Iter " << iter << " gbest_fitness = " << gbest_fitness << endl;
        }
    }

    // decode gbest to final state and compute summary
    vector<vector<int>> final_state;
    bool final_feas;
    dd final_fitness = fitnessOfBinaryParticle(gbest, final_state, final_feas);
    ResultSummary result = evaluateSchedule(final_state);

    // Save schedule_output.csv
    ofstream out("schedule_output.csv");
    out << "Slot,Time,VehicleID,State\n";
    for (int t=0;t<NT;++t) {
        string timestr = slotToTimeStr(t);
        for (int i=0;i<N;++i) {
            out << t << "," << timestr << "," << vehicles[i].id << "," << final_state[i][t] << "\n";
        }
    }
    out.close();
    // Save grid profile
    ofstream gfile("grid_profile.csv");
    gfile << "Slot,Time,P_grid_kW,Price_Rs_per_kWh,Cost_Rs\n";
    for (int t=0;t<NT;++t) {
        dd cost = result.P_grid[t] * price_per_slot[t] * DT;
        gfile << t << "," << slotToTimeStr(t) << "," << result.P_grid[t] << "," << price_per_slot[t] << "," << cost << "\n";
    }
    gfile.close();

    // print summary
    cout << "----- Optimization Summary -----\n";
    cout << "Final fitness: " << final_fitness << "\n";
    cout << "Total grid cost (optimized): " << result.total_cost << " Rs\n";
    cout << "Peak grid draw (optimized): " << result.peak << " kW\n";
    cout << "Feasible (no heavy penalties)? " << (final_feas ? "Yes" : "No (penalties applied)") << "\n";
    cout << "Schedule and grid profile saved to schedule_output.csv and grid_profile.csv\n";
}

// ---------- baseline naive schedule: charge on arrival until req met ----------
ResultSummary baselineNaive() {
    int NT = SLOTS_PER_DAY;
    vector<vector<int>> state(N, vector<int>(NT, 0));
    // for each vehicle, greedily charge from arrival until SOC_req or departure reached
    for (int i=0;i<N;++i) {
        dd soc = vehicles[i].soc_init;
        for (int t=vehicles[i].arrival_slot; t<vehicles[i].departure_slot; ++t) {
            if (soc + 1e-9 >= vehicles[i].soc_req) break;
            // charge
            state[i][t] = 1;
            dd deltaE = vehicles[i].p_ch_max * ETA_CH * DT;
            soc += deltaE / vehicles[i].capacity_kWh;
        }
    }
    return evaluateSchedule(state);
}

// ---------- main ----------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    string fname = "vehicles.csv";
    if (!loadVehicles(fname)) {
        cerr << "Failed to load vehicles.csv. Ensure file exists and format is correct.\n";
        return 1;
    }
    // optional: try load price.csv
    ifstream pf("price.csv");
    if (pf.good()) {
        cout << "price.csv found - loading custom price profile\n";
        tryLoadPrice("price.csv");
    } else {
        // default TOU price: base 5 Rs/kWh, peak 20 Rs/kWh between 17:00-21:00
        for (int t=0;t<SLOTS_PER_DAY;++t) price_per_slot[t]=5.0;
        int peakStart = timeToSlot(17.0), peakEnd = timeToSlot(21.0);
        for (int t=peakStart;t<=peakEnd && t<SLOTS_PER_DAY;++t) price_per_slot[t] = 20.0;
    }

    cout << "Loaded " << N << " vehicles.\n";

    // Show baseline
    ResultSummary base = baselineNaive();
    cout << "Baseline: total cost = " << base.total_cost << " Rs, peak = " << base.peak << " kW\n";

    // Run BPSO
    runBPSO();

    return 0;
}
