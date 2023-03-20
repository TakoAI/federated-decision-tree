# Federated Decision Tree

In this repository, we implemented a method to MERGE 2 decision trees.

The merge function provides the scalability to scale decision tree training into hundreds or more machines and then merge them back into a reasonable size.

With this approach, we developed the federated decision tree with about the same performance as boosted trees, while maintaining a size upper bound with feature number. This greatly increase the scalability of training decision trees on separated machines.
