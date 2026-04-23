#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    matrix_memory_allocator.Bind(current_query, "query_round_" + std::to_string(i));

    gpu_sim.MoveMatrixToSharedMem(current_query);

    Matrix *k_stack = matrix_memory_allocator.Allocate("k_stack_init");
    Matrix *v_stack = matrix_memory_allocator.Allocate("v_stack_init");

    // Build stacked K and V in SRAM (rows: 0..i)
    for (size_t j = 0; j <= i; ++j) {
      matrix_memory_allocator.Bind(keys[j], "key_" + std::to_string(j));
      matrix_memory_allocator.Bind(values[j], "value_" + std::to_string(j));
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      if (j == 0) {
        gpu_sim.Copy(keys[j], k_stack, kInSharedMemory);
        gpu_sim.Copy(values[j], v_stack, kInSharedMemory);
      } else {
        Matrix *k_next = matrix_memory_allocator.Allocate("k_stack_next_" + std::to_string(j));
        Matrix *v_next = matrix_memory_allocator.Allocate("v_stack_next_" + std::to_string(j));
        gpu_sim.Concat(k_stack, keys[j], k_next, 0, kInSharedMemory);
        gpu_sim.Concat(v_stack, values[j], v_next, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(k_stack);
        gpu_sim.ReleaseMatrix(v_stack);
        k_stack = k_next;
        v_stack = v_next;
      }
    }

    // Transpose K in place to get K^T
    gpu_sim.Transpose(k_stack, kInSharedMemory);

    // Scores = Q * K^T
    Matrix *scores = matrix_memory_allocator.Allocate("scores");
    gpu_sim.MatMul(current_query, k_stack, scores);

    // Prepare values in SRAM
    for (size_t j = 0; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
    }

    // Build answer row by row: for each row softmax weights, accumulate scaled V[j]
    Matrix *answer = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(scores, r, row, kInSharedMemory);
      Matrix *row_exp = matrix_memory_allocator.Allocate("row_exp_" + std::to_string(r));
      gpu_sim.MatExp(row, row_exp);
      Matrix *sum_row = matrix_memory_allocator.Allocate("sum_row_" + std::to_string(r));
      gpu_sim.Sum(row_exp, sum_row);
      Matrix *row_soft = matrix_memory_allocator.Allocate("row_soft_" + std::to_string(r));
      gpu_sim.MatDiv(row_exp, sum_row, row_soft);

      Matrix *row_acc = nullptr;
      for (size_t j = 0; j <= i; ++j) {
        Matrix *w = matrix_memory_allocator.Allocate("w_" + std::to_string(r) + "_" + std::to_string(j));
        gpu_sim.GetColumn(row_soft, j, w, kInSharedMemory);
        Matrix *scaled_v = matrix_memory_allocator.Allocate("scaled_v_" + std::to_string(r) + "_" + std::to_string(j));
        gpu_sim.MatMulNum(values[j], w, scaled_v);
        if (j == 0) {
          row_acc = matrix_memory_allocator.Allocate("row_acc_" + std::to_string(r));
          gpu_sim.Copy(scaled_v, row_acc, kInSharedMemory);
        } else {
          Matrix *row_next = matrix_memory_allocator.Allocate("row_next_" + std::to_string(r) + "_" + std::to_string(j));
          gpu_sim.MatAdd(row_acc, scaled_v, row_next);
          gpu_sim.ReleaseMatrix(row_acc);
          row_acc = row_next;
        }
        gpu_sim.ReleaseMatrix(w);
        gpu_sim.ReleaseMatrix(scaled_v);
      }

      if (r == 0) {
        answer = matrix_memory_allocator.Allocate("answer_round_" + std::to_string(i));
        gpu_sim.Copy(row_acc, answer, kInSharedMemory);
      } else {
        Matrix *ans_next = matrix_memory_allocator.Allocate("answer_next_" + std::to_string(r));
        gpu_sim.Concat(answer, row_acc, ans_next, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(answer);
        answer = ans_next;
      }

      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_exp);
      gpu_sim.ReleaseMatrix(sum_row);
      gpu_sim.ReleaseMatrix(row_soft);
      gpu_sim.ReleaseMatrix(row_acc);
    }

    // Move answer to HBM for committing
    gpu_sim.MoveMatrixToGpuHbm(answer);

    // Release intermediates
    gpu_sim.ReleaseMatrix(scores);
    gpu_sim.ReleaseMatrix(k_stack);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
