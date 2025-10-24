#include <iostream>
#include <random>
using std::cout;
using std::endl;
using std::uniform_int_distribution;
using std::mt19937;

void MergeSortL(int *array, int size);

void MergeL(int *temp, int *part1, int *part2, int len1, int len2);

int FindPivot(int *array, int left, int right);

void PartitionL(int *array, int left, int right);

void QuickSort(int *array, int left, int right);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
int counter;
// 1.1
void MergeSortL(int *array, int size) {
  int *temp = new int[size];
  for (int part_size = 1; part_size < size; part_size *= 2) {
    for (int j = 0; j + part_size < size; j += part_size * 2) {
      int part2_len = part_size;
      part2_len = std::min(part_size, size - j - part_size);
      MergeL(temp, &array[j], &array[j + part_size], part_size, part2_len);
      for (int i = 0; i < part_size + part2_len; i++)
        array[j + i] = temp[i];
    }
  }
  delete[] temp;
}

void MergeL(int temp[], int *part1, int *part2, int len1, int len2) {
  int i = 0, j = 0, k = 0;
  while (i < len1 && j < len2) {
	  counter++;
    if (part1[i] <= part2[j]) {
      temp[k] = part1[i++];
    } else {
      temp[k] = part2[j++];
    }
    // Test
    counter++;
    // End Test
    k++;
  }
  while (i < len1) {
    temp[k++] = part1[i++];
    counter++;
  }
  while (j < len2) {
    temp[k++] = part2[j++];
    counter++;
  }
}

// 1.1
void QuickSort(int *array, int left, int right) {
  if (left >= right)
    return;
  int p;
  p = FindPivot(array, left, right);
  QuickSort(array, left, p - 1);
  QuickSort(array, p + 1, right);
}

int FindPivot(int *array, int left, int right) {
  int p = array[left], i = left + 1, j = right;
  while (i <= j) {
    while (array[i] <= p && i <= j)
      i++;
    while (p <= array[j] && i <= j)
      j--;

    if (i < j) {
      std::swap(array[i], array[j]);
      counter++;
    }
  }
  counter ++;
  std::swap(array[left], array[j]);
  return j;
}

// 2.2
void BinaryInSort(int *array, int size) {
  for (int i = 1; i < size; i++) {
    int x = array[i];
    int l = 0, r = i - 1;
    int m;
    while (l <= r) {
      m = (l + r) / 2;

      if (array[m] < x)
        l = m + 1;
      else
        r = m - 1;
    }
    for (m = i - 1; m >= l; m--) {
      array[m + 1] = array[m];
      counter ++;
    }
    array[l] = x;
    counter++;
  }
}

// 2.2
void ReassignArray(int *array, int size) {
  int l = 0, r = size - 1;
  while (l <= r) {
    while (l <= r && array[l] < 0)
      l++;
    while (l <= r && array[r] > 0)
      r--;
    if (l < r) {
      std::swap(array[l], array[r]);
    }
  }
}

// 2.3
void Hanoi(int n, char from, char help, char to) {
  // 单个移动：把第n个盘子移动到to
  if (n == 1) {
    std::cout << "Move #" << n << " from " << from << " to " << to << std::endl;
    return;
  }
  // 递归移动：先把上面n-1个盘子移动到help
  Hanoi(n - 1, from, to, help);
  // 单个移动：把第n个盘子移动到to
  std::cout << "Move #" << n << " from " << from << " to " << to << std::endl;

  // 递归移动：把help上的n-1个盘子移动到to
  Hanoi(n - 1, help, from, to);
}

// 2.4
int BinarySearchTop(int *array, int size) {
  // 类似于二分查找数值，这次是查找一个pattern 满足 x1<x, x>x2
  int l = 0, r = size - 1, m;
  while (l <= r) {
    m = (l + r) / 2;
    // on edge, exit
    if (m == l || m == r)
      return -1;
    // if pattern
    if (array[m - 1] < array[m] && array[m] < array[m + 1]) {
      // on left side
      l = m;
    } else if (array[m - 1] > array[m] && array[m] > array[m + 1]) {
      // on right side
      r = m;
    } else
      return m;
  }
  return -1;
}



int BinarySearchOnHighNoExist(int *array, int size, int k) {
	// find first number x >= k (Right Edge)
	int l=0, r=size-1, m;
	while (l<=r) {
		m=(l+r)/2;
		if (array[m] >= k) {
			r=m - 1;
		} else if (array[m] < k) {
			l=m+1;
		}
	}
	return l;
}

int BinarySearchOnLowNoExist(int *array, int size, int k) {
	// find first number x > k (Left Edge)
	int l=0, r=size-1, m;
	while (l<=r) {
		m=(l+r)/2;
		if (array[m] > k) {
			r=m - 1;
		} else if (array[m] <= k) {
			l=m+1;
		}
	}
	return l;
}

void SelectRange(int *array, int size, int low, int high) {
	// O(logn)
	int lowInd= BinarySearchOnLowNoExist(array, size, low);
	// O(logn)
	int highInd= BinarySearchOnHighNoExist(array, size, high);
	// find them!
	for (int i=lowInd; i<highInd; i++) {
		std::cout << array[i] << ' ';
	}
	std::cout << std::endl;
}



int main() {
  int sort_data[20] = {2,  1,  7,  9,  6,  4,  8,  5,  3,  10,
                       12, 11, 14, 13, 16, 15, 19, 17, 18, 0};
  // 只有正数和负数
  int data2[20] = {2, -1, 1, -7, 7,  9,   -3, 6, -5, 4,
                   8, -2, 5, 3,  10, -10, -4, 9, -6, -11};

  int peakData[20] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                      9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

  int peakData2[20] = {9, 10, 9,  8,  7,  6,  5,  4,  3,  2,
                       1, 0,  -1, -2, -3, -4, -5, -6, -7, -8};

  // for (int i = 0; i < 20; i++) {
  //   cout << data2[i] << " ";
  // }
  // cout << endl;

  // MergeSortL(sort_data, 20);
  // QuickSort(sort_data, 0, 19);
  // BinaryInSort(sort_data, 20);
  // ReassignArray(data2, 20);
  // Hanoi(3, 'A', 'B', 'C');
  /*
  int solveLength[] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,10000};

  for (int i=0; i< sizeof(solveLength)/ sizeof(int); i++) {
	  int *test_arr = new int[solveLength[i]];
	  // Gen data
	  std:: mt19937 gen(100);
	  for (int j=0; j<solveLength[i]; j++) {
		  std::uniform_int_distribution<int> dist(0, solveLength[i]);
		test_arr[j] = dist(gen);
	  }

	  counter = 0;
	  //MergeSortL(test_arr, solveLength[i]);
	  QuickSort(test_arr, 0, solveLength[i]-1);
	  delete[] test_arr;
	  // output
	  std::cout << "MergeSort: size=" << solveLength[i] << " tick=" << counter << std::endl;
  }
  std::cout << "top is at #" << BinarySearchTop(peakData, 20) << endl;
*/
  int rangeData[]={1,2,3,4,5,6,7,8,9,10};

  SelectRange(rangeData,10, 3, 9);

  // for (int i = 0; i < 20; i++) {
  //   cout << data2[i] << " ";
  // }
  cout << endl;
  return 0;
}
