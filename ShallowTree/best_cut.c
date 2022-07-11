#ifndef LIB
#define LIB

#ifndef BEST_CUT
#define BEST_CUT

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

double *array;

int compare(const void *a, const void *b) {
  int ia = *(int *) a;
  int ib = *(int *) b;
  int ans = 0;
  if (array[ia] < array[ib]) {
    ans = -1;
  }
  else if (array[ia] > array[ib]) {
    ans = 1;
  }
  return ans;
}

int *order_data(double *data, int n) {
  int i;
    int *ans = (int *)malloc(sizeof(int) * n);
  for (i = 0; i < n; i++) {
    ans[i] = i;
  }
  array = data;
  qsort(ans, n, sizeof(*ans), compare);
  return ans;
}

// returns the numerator of the weighted height. At the end, it must 
// be divided by N
double get_height_cost(int curr_height,int N, int K, double alpha, 
                       double beta){
  int N_l, N_r,K_l,K_r;
  if (K==1){
    return N*curr_height;
  }

  K_l = (int)ceil(K*beta);
  K_l = (K_l>1)?K_l:1;
  K_l = (K_l<K)?K_l:K-1;
  K_r = K - K_l;
  if(N==1){
    if(K_l>K_r) N_l=1;
    else N_l =0;
  }
  else{
    N_l = (int)ceil(N*alpha);
    N_l = (N_l>1)? N_l:1;
    N_l = (N_l<N)? N_l:N-1;
  }


  N_r = N - N_l;

  return get_height_cost(curr_height+1,N_l,K_l,alpha,beta)+
         get_height_cost(curr_height+1,N_r,K_r,alpha,beta);

}

double get_cur_height_cost(int N_lAux, int N_rAux,
                           int K_lAux, int K_rAux,
                           int n,
                           float alpha, float beta,
                           bool cut_left, bool cut_right) {
  double ans = 0;
  double height_cost_left = get_height_cost(1,N_lAux,K_lAux,alpha,beta);
  double height_cost_right = get_height_cost(1,N_rAux,K_rAux,alpha,beta);
  if (cut_left) {
    height_cost_left -= (double)N_lAux;
  }
  if (cut_right) {
    height_cost_right -= (double)N_rAux;
  }
  ans = (height_cost_left + height_cost_right)/n;
  return ans;
}

void best_cut_single_dim(double *data, int *data_count, double *centers,
                         double *distances, int *dist_order, int n, int k,
                         double *ans, double height_factor,
                         bool cut_left, bool cut_right) {
  int i, j, c, cur_c, ix, ic;
  int idx_data = 0;
  int idx_centers = 0;
  int N_l=0;
  int N_r =0;
  int K_r =0;
  int K_l =0;
  int N_lAux,N_rAux,K_lAux,K_rAux;
  double nxt_cut;
  double best_cut;
  double alpha;
  double beta;
  double cur_cost = 0;
  double init_dist_cost=0;
  double cur_dist_cost =0;
  double cur_height_cost =0;
  double best_cost;
  double old_data_cost;
  double max_cut;
  int *left_data_mask = (int *)malloc(sizeof(int) * n);
  int *left_centers_mask = (int *)malloc(sizeof(int) * k);
  int *best_in_left = (int *)malloc(sizeof(int) * n);
  int *best_in_right = (int *)malloc(sizeof(int) * n);
  double *cur_dist_costs = (double *)malloc(sizeof(double) * n);
  int *cur_centers = (int *)malloc(sizeof(int) * n);
  int *data_order = (int *)malloc(sizeof(int) * n);
  int *centers_order = (int *)malloc(sizeof(int) * k);
  bool valid = false;

  // data order array
  for (i = 0; i < n; i++) {
    data_order[i] = i;
  }
  array = data;
  qsort(data_order, n, sizeof(*data_order), compare);

  // centers order array
  for (i = 0; i < k; i++) {
    centers_order[i] = i;
  }
  array = centers;
  qsort(centers_order, k, sizeof(*centers_order), compare);
  
  // next cut is the smallest cut
  // max cut is largest value of center
  c = centers_order[0];
  nxt_cut = centers[c];
  c = centers_order[k-1];
  max_cut = centers[c];

  // current costs are the best costs (no cuts yet)
  for (i = 0; i < n; i++) {
    c = dist_order[i * k];
    cur_centers[i] = c;
    cur_dist_costs[i] = distances[i*k + c] * data_count[i];
    cur_dist_cost += cur_dist_costs[i];
  }
  init_dist_cost = cur_dist_cost;
  // consider a single center is in the left
  c = centers_order[0];

  // vector to keep track of best center to the left
  // initially, the best center to the left is the only one there
  for (i = 0; i < n; i++) {
    best_in_left[i] = c;
  }

  // vector to keep track of index of centers that
  // have been tested to the right
  // initially, the best center to the right is the first one
  for (i = 0; i < n; i++) {
    best_in_right[i] = 0;
  }

  // set indices of data and center as first index in
  // which the element is to the right of the cut
  ix = data_order[idx_data];
  while ( (data[ix] <= nxt_cut) && idx_data < n) {
    idx_data++;
    if (idx_data < n) {
      ix = data_order[idx_data];
    }
  }
  
  while ( (centers[c] <= nxt_cut) && (idx_centers < k) ) {
    idx_centers++;
    if (idx_centers < k) {
      c = centers_order[idx_centers];
      // if a center is moved to the left during initialization,
      // check if it's the best center to the left for each datum
      if (centers[c] <= nxt_cut) {
        for (i = 0; i < n; i++) {
          cur_c = best_in_left[i];
          if (distances[i*k + cur_c] > distances[i*k + c]) {
            best_in_left[i] = c;
          }
        }
      }
    }
  }
  
  // if all centers moved to the left after the
  // first cut, return (no cut is possible)
  if (idx_centers == k) {
    ans[0] = -1;
    ans[1] = INFINITY;
    free(cur_dist_costs);
    free(left_data_mask);
    free(left_centers_mask);
    free(best_in_left);
    free(best_in_right);
    free(cur_centers);
    free(data_order);
    free(centers_order);
    return;
  }

  // define masks
  for (i = 0; i < n; i++) {
    left_data_mask[i] = data[i] <= nxt_cut;
    if(left_data_mask[i]) N_l++;
    else N_r++;
  }

  for (i = 0; i < k; i++) {
    left_centers_mask[i] = centers[i] <= nxt_cut;
    if(left_centers_mask[i]) K_l++;
    else K_r++;
  }

  // reassign data separated from their centers
  for (i = 0; i < n; i++) {
    cur_c = cur_centers[i];
    // if datum is to the left, best center is in best_in_left vector
    if (left_data_mask[i]) {

      c = best_in_left[i];
    }
    // if datum is to the right, find center to the right closest to datum
    // (going from the best_in_right vector)
    else {
      j = best_in_right[i];
      c = dist_order[i*k + j];
      while (left_centers_mask[c]) {
        j++;
        c = dist_order[i*k + j];
      }
      best_in_right[i] = j;
    }
    // if best center is different than current center, update cost
    if (c != cur_c) {
      old_data_cost = cur_dist_costs[i];
      cur_dist_costs[i] = distances[i*k + c] * data_count[i];
      cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
      cur_centers[i] = c;
    }
  }

  N_lAux = (N_l>1)? N_l:1;
  N_lAux = (N_lAux==n)? n-1:N_lAux;
  N_rAux = n - N_lAux;
  K_lAux = (K_l>1)?K_l:1;
  K_lAux = (K_lAux==k)?k-1:K_lAux;
  K_rAux = k - K_lAux;
  alpha = (double)N_lAux/n;
  beta = (double) K_lAux/k;
  
  cur_height_cost = get_cur_height_cost(N_lAux, N_rAux, K_lAux, K_rAux,
                                        n, alpha, beta, cut_left, cut_right);
  cur_cost = cur_dist_cost/init_dist_cost+height_factor*cur_height_cost;
  
  // store initial cut and cost as best ones (if they're possible)
  if ((idx_centers != 0) && (idx_centers != k) &&
      (idx_data >= idx_centers) && 
      ( (n - idx_data) >= (k - idx_centers) ) ) {
       best_cut = nxt_cut;
       best_cost = cur_cost;
  }
  else {
    best_cut = -1;
    best_cost = INFINITY;
  }
  
  while ( (idx_data < n) && (idx_centers < k) ) {
    // find next cut and check if feasible
    ix = data_order[idx_data];
    ic = centers_order[idx_centers];
    if (data[ix] < centers[ic]) {
      nxt_cut = data[ix];
    }
    else {
      nxt_cut = centers[ic];
    }
    if (nxt_cut >= max_cut) {
      break;
    }

    // move data points to the left and assign them to best center 
    // to the left
    while ( (idx_data < n) && (data[ix] <= nxt_cut) ) {
      old_data_cost = cur_dist_costs[ix];
      left_data_mask[ix] = 1;
      N_l++;
      N_r--;
      // find best center to the left via best_in_left vector
      c = best_in_left[ix];
      cur_centers[ix] = c;
      cur_dist_costs[ix] = distances[ix*k + c] * data_count[ix];
      cur_dist_cost += (cur_dist_costs[ix] - old_data_cost);
      idx_data++;
      if (idx_data < n) {
          ix = data_order[idx_data];
      }
    }

    // move centers to the left and reassign data points
    while ( (idx_centers < k) && (centers[ic] <= nxt_cut) ) {
      left_centers_mask[ic] = 1;
      K_l++;
      K_r--;
      for (i = 0; i < n; i++) {
        old_data_cost = cur_dist_costs[i];
        // update best_in_left vector
        cur_c = best_in_left[i];
        if (distances[i*k + ic] < distances[i*k + cur_c]) {
          best_in_left[i] = ic;
        }
        // if datum is to the left, assign it to center that moved 
        // to the left if it's best in left
        if (left_data_mask[i]) {
          if (best_in_left[i] == ic) {
            cur_centers[i] = ic;
            cur_dist_costs[i] = distances[i*k + ic] * data_count[i];
            cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
          }
        }
        // if datum is to the right and its current center moved left, find
        // new best center (starting search from index in best_in_right)
        else if (cur_centers[i] == ic) {
          j = best_in_right[i];
          c = dist_order[i*k];
          while (left_centers_mask[c]) {
            j++;
            c = dist_order[i*k + j];
          }
          best_in_right[i] = j;
          cur_dist_costs[i] = distances[i*k + c] * data_count[i];
          cur_dist_cost += (cur_dist_costs[i] - old_data_cost);
          cur_centers[i] = c;
        }
      }
      idx_centers++;

      if (idx_centers < k) {
        ic = centers_order[idx_centers];
      }
    }

  N_lAux = (N_l>1)? N_l:1;
  N_lAux = (N_lAux>n)? n-1:N_lAux;
  N_rAux = n - N_lAux;
  K_lAux = (K_l>1)?K_l:1;
  K_lAux = (K_lAux>k)?k-1:K_lAux;
  K_rAux = k - K_lAux;
  alpha = (double)N_lAux/n;
  beta = (double) K_lAux/k;

  if ((idx_centers != 0) && (idx_centers != k) && (idx_data >= idx_centers) 
      && ( (n - idx_data) >= (k - idx_centers) )  ) {
    //check if cut is valid: must have at least as many centers
    // as data points on each side
    valid = true;
    N_lAux = (N_l>1)? N_l:1;
    N_lAux = (N_lAux>n)? n-1:N_lAux;
    N_rAux = n - N_lAux;
    K_lAux = (K_l>1)?K_l:1;
    K_lAux = (K_lAux>k)?k-1:K_lAux;
    K_rAux = k - K_lAux;
    alpha = (double)N_lAux/n;
    beta = (double) K_lAux/k;

    if(K_lAux<=0 || K_rAux<=0){
      cur_height_cost = 100000;
      printf("weird case \n");
    }
    else{
      cur_height_cost = get_cur_height_cost(N_lAux, N_rAux, K_lAux, K_rAux,
                                            n, alpha, beta, cut_left, 
                                            cut_right);
    }
    cur_cost = cur_dist_cost/init_dist_cost+height_factor*cur_height_cost;
  }
  else{
    valid = false;

  }

if(valid){
  // check if cut is acceptable: must have at least as many centers
    // as data points on each side
    if (valid  && (cur_cost < best_cost) ) {
         best_cut = nxt_cut;
         best_cost = cur_cost;
    }
}
}

  // return best cut and cost
  ans[0] = best_cut;
  ans[1] = best_cost;

  free(cur_dist_costs);
  free(left_data_mask);
  free(left_centers_mask);
  free(best_in_left);
  free(best_in_right);
  free(cur_centers);
  free(data_order);
  free(centers_order);
  return;
};

#endif


int main () {}

#endif
