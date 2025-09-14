import time
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes
from typing import List, Tuple
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from flwr.server import ServerConfig


class CustomStrategy(FedAvg):
    def __init__(self, *args, pushgateway_address="pushgateway:9091", **kwargs):
        super().__init__(*args, **kwargs)
        self.pushgateway_address = pushgateway_address

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config["server_round"] = server_round
        return instructions

    def aggregate_fit(self, rnd, results, failures):
        round_duration = time.time() - self.round_start_time
        self.push_metrics(rnd, round_duration, len(results))
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> float:
        total_loss = 0.0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1_macro = 0.0
        total_f1_weighted = 0.0
        total_samples = 0
        start_time = time.time()

        for client, eval_res in results:
            num_examples = eval_res.num_examples
            metrics = eval_res.metrics

            loss = eval_res.loss
            accuracy = metrics.get("accuracy", 0.0)
            precision = metrics.get("precision", 0.0)
            recall = metrics.get("recall", 0.0)
            f1_macro = metrics.get("f1_macro", 0.0)
            f1_weighted = metrics.get("f1_weighted", 0.0)

            total_loss += loss * num_examples
            total_accuracy += accuracy * num_examples
            total_precision += precision * num_examples
            total_recall += recall * num_examples
            total_f1_macro += f1_macro * num_examples
            total_f1_weighted += f1_weighted * num_examples
            total_samples += num_examples

        eval_duration = time.time() - start_time

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
        avg_precision = total_precision / total_samples if total_samples > 0 else 0.0
        avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
        avg_f1_macro = total_f1_macro / total_samples if total_samples > 0 else 0.0
        avg_f1_weighted = total_f1_weighted / total_samples if total_samples > 0 else 0.0

        self.push_eval_metrics(
            rnd, avg_loss, avg_accuracy, eval_duration,
            avg_recall, avg_precision, avg_f1_macro, avg_f1_weighted
        )

        return super().aggregate_evaluate(rnd, results, failures)

    def push_metrics(self, round_num, round_duration, num_clients,
                     recall=0.0, precision=0.0, f1_macro=0.0, f1_weighted=0.0):
        registry = CollectorRegistry()
        g_round = Gauge('fl_rounds_total', 'Total number of training rounds', registry=registry)
        g_clients = Gauge('fl_connected_clients', 'Number of connected clients', registry=registry)
        g_duration = Gauge('fl_round_duration_seconds', 'Training round duration (s)', registry=registry)
        g_recall = Gauge('recall', 'Recall', registry=registry)
        g_precision = Gauge('precision', 'Precision', registry=registry)
        g_f1_macro = Gauge('f1_macro', 'F1 Score Macro', registry=registry)
        g_f1_weighted = Gauge('f1_weighted', 'F1 Score Weighted', registry=registry)

        g_round.set(round_num)
        g_clients.set(num_clients)
        g_duration.set(round_duration)
        g_recall.set(recall)
        g_precision.set(precision)
        g_f1_macro.set(f1_macro)
        g_f1_weighted.set(f1_weighted)

        push_to_gateway(self.pushgateway_address, job="fl_server", registry=registry)

    def push_eval_metrics(self, round_num, loss, accuracy, eval_time,
                          recall, precision, f1_macro, f1_weighted):
        registry = CollectorRegistry()
        g_loss = Gauge('fl_loss', 'Global aggregated loss', registry=registry)
        g_acc = Gauge('fl_accuracy', 'Global aggregated accuracy', registry=registry)
        g_eval_time = Gauge('fl_evaluation_duration_seconds', 'Evaluation time in seconds', registry=registry)
        g_recall = Gauge('recall', 'Recall', registry=registry)
        g_precision = Gauge('precision', 'Precision', registry=registry)
        g_f1_macro = Gauge('f1_macro', 'F1 Score Macro', registry=registry)
        g_f1_weighted = Gauge('f1_weighted', 'F1 Score Weighted', registry=registry)

        g_loss.set(loss)
        g_acc.set(accuracy)
        g_eval_time.set(eval_time)
        g_recall.set(recall)
        g_precision.set(precision)
        g_f1_macro.set(f1_macro)
        g_f1_weighted.set(f1_weighted)

        push_to_gateway(self.pushgateway_address, job="fl_server",
                        grouping_key={'round': f"{round_num:02}"}, registry=registry)


if __name__ == "__main__":
    strategy = CustomStrategy(pushgateway_address="pushgateway:9091")
    config = ServerConfig(num_rounds=30)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=config,
    )
