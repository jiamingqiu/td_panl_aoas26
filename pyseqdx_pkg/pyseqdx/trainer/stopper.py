from typing import Optional, List, Union, Tuple
import torch


# stopper function for training process
class TrainStopper:
    def __init__(
        self,
        # hard stop
        min_epoch=0,
        max_epoch=5,
        # soft/early stop, when loss stop decreasing
        patience=1,
        min_buffer_rel=0,
        # non-zero -> stop with stablized loss (not necessarily decreasing)
        min_delta_rel=0,
        min_delta_abs=0,
        stop_mode="noimprove",
        # misc
        loss_no_more_than=float("inf"),
        **kwargs,
    ):

        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.loss_no_more_than = loss_no_more_than

        self.patience = patience
        self.min_buffer_rel = min_buffer_rel

        self.min_delta_rel = min_delta_rel
        self.min_delta_abs = min_delta_abs

        assert stop_mode in ["nochange", "noimprove", "num_epoch"]
        self.stop_mode = stop_mode
        if self.stop_mode == "nochange":
            assert (
                self.min_delta_abs > 0 and self.min_delta_rel > 0
            ), "min_delta cannot be zero"
        if self.stop_mode == "num_epoch":
            assert (
                self.min_epoch == self.max_epoch
            ), f"min_epoch, max_epoch need to be identical."

        if min_epoch == max_epoch:
            self.need_validation_loss = False
        else:
            self.need_validation_loss = True

        self.reset()

    @classmethod
    def by_num_epochs(cls, num_epochs):
        return cls(min_epoch=num_epochs, max_epoch=num_epochs, stop_mode="num_epoch")

    @classmethod
    def by_noimprove(
        cls,
        min_epoch=0,
        max_epoch=5,
        patience=1,
        min_buffer_rel=0.05,
        loss_no_more_than=float("inf"),
    ):
        # stop if (no improvement more than patience epoch) or (significant worsen)
        # no improvement defined by current_loss > known_min
        # significant worsen defined by current_loss > know_min + abs(know_min) * min_buffer_rel
        # hopefully this makes the stopper more patient
        stopper = cls(
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            patience=patience,
            min_buffer_rel=min_buffer_rel,
            min_delta_rel=0,
            min_delta_abs=0,
            stop_mode="noimprove",
            loss_no_more_than=loss_no_more_than,
        )
        return stopper

    @classmethod
    def by_nochange(
        cls,
        min_epoch=0,
        max_epoch=5,
        patience=1,
        loss_no_more_than=float("inf"),
        min_delta_rel=0.01,
        min_delta_abs=0.005,
    ):
        stopper = cls(
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            patience=patience,
            loss_no_more_than=loss_no_more_than,
            min_delta_rel=min_delta_rel,
            min_delta_abs=min_delta_abs,
            stop_mode="nochange",
        )
        return stopper

    def reset(self):
        self.epoch = 1
        self.counter = 0
        self.min_validation_loss = None
        self.previous_validation_loss = None

    def stop(
        self, validation_loss: Union[float, Tuple[float, ...], List[float]]
    ) -> bool:
        if self.epoch >= self.max_epoch:
            return True
        else:
            self.epoch += 1
        if self.epoch <= self.min_epoch:
            return False
        # Normalize to tuple for consistent processing
        if isinstance(validation_loss, (float, int)):
            validation_loss = (float(validation_loss),)
        else:
            validation_loss = tuple(validation_loss)

        # Early rejection: if any component exceeds allowed max loss
        if any(loss > self.loss_no_more_than for loss in validation_loss):
            return False

        if self.stop_mode == "noimprove":
            return self.stop_noimprove(validation_loss)
        elif self.stop_mode == "nochange":
            return self.stop_nochange(validation_loss)
        else:
            raise ValueError(f"Unknown stop mode {self.stop_mode}")

        # if self.epoch >= self.max_epoch:
        #     return True
        # else:
        #     self.epoch += 1

        # if self.epoch <= self.min_epoch:
        #     return False
        # if validation_loss > self.loss_no_more_than:
        #     return False

        # # get here only if (small validation loss & more that minepoch)
        # if validation_loss < self.min_validation_loss:
        #     self.min_validation_loss = validation_loss
        #     self.counter = 0
        # else:
        #     self.counter += 1
        #     if self.counter >= self.patience:
        #         return True
        # return False

    def stop_nochange(
        self, validation_loss: Union[float, Tuple[float, ...], List[float]]
    ) -> bool:

        # Initialize previous_validation_loss on first call
        if self.previous_validation_loss is None:
            self.previous_validation_loss = list(validation_loss)
            self.counter = 0
            return False

        # check if any of the losses is not stablized
        changing = False
        for i, loss in enumerate(validation_loss):
            if self.is_meaningful_change(loss, self.previous_validation_loss[i]):
                changing = True
            self.previous_validation_loss[i] = loss

        if changing:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False

    def is_meaningful_change(self, current: float, reference: float) -> bool:
        abs_change = abs(current - reference)
        # if self.mode == "absolute":
        #     return abs_change >= self.min_delta
        # elif self.mode == "relative":
        #     return abs_change >= self.min_delta * max(abs(reference), self.epsilon)
        # elif self.mode == "hybrid":  # in case reference is 0
        abs_thresh = self.min_delta_abs
        rel_thresh = self.min_delta_rel * abs(
            reference
        )  # max(abs(reference), self.epsilon)
        return abs_change >= max(abs_thresh, rel_thresh)

    def stop_noimprove(
        self, validation_loss: Union[float, Tuple[float, ...], List[float]]
    ) -> bool:

        # Initialize min_validation_loss on first call
        if self.min_validation_loss is None:
            self.min_validation_loss = list(validation_loss)
            self.counter = 0
            return False
        # check if any of the losses improved
        improved = False
        significant_worse = False
        for i, loss in enumerate(validation_loss):
            known_min = self.min_validation_loss[i]
            if loss < known_min:
                self.min_validation_loss[i] = loss
                improved = True
            elif loss > known_min + abs(known_min) * self.min_buffer_rel:
                significant_worse = True

        if improved:
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience or significant_worse:
            return True

        return False


# special stopper for NuLaga cycle only
class TrainNuLagaStopper:
    def __init__(
        self,
        model,
        desired_tpr,
        desired_cost,
        # validation data
        loader_validate,
        # hard stop
        min_epoch=0,
        max_epoch=5,
        # soft/early stop, when desired performance reached (max abs err)
        validate_epoch=1,  # validate performance every ? epochs
        desired_tol=1e-2,
        patience=1,
        lagdual_stopper=TrainStopper.by_num_epochs(1),  # stopper on lagdual
        laga_deadzone=0.075,
        **kwargs,
    ):
        (
            self.model,
            self.desired_tpr,
            self.desired_cost,
            self.loader_validate,
            self.min_epoch,
            self.max_epoch,
            self.validate_epoch,
            self.desired_tol,
            self.patience,
            self.lagdual_stopper,
        ) = (
            model,
            desired_tpr,
            desired_cost,
            loader_validate,
            min_epoch,
            max_epoch,
            validate_epoch,
            desired_tol,
            patience,
            lagdual_stopper,
        )

        # waive desired requirement if laga within deadzone
        self.laga_deadzone = laga_deadzone

        # for compatibility
        self.need_validation_loss = True

        self.reset()

    def reset(self):
        self.epoch = 1
        self.counter = 0

    def stop(
        self,
        validation_loss: Union[float, Tuple[float, ...], List[float]],
        *args,
        **kwargs,
    ):
        # validation_loss passed to lagdual_stopper

        if self.epoch >= self.max_epoch:
            return True
        else:
            self.epoch += 1
        if self.epoch <= self.min_epoch:
            return False

        lagdual_stop = self.lagdual_stopper.stop(validation_loss)

        performance_stop = [False, False]
        # waive desired requirement if any laga too close to zero.
        current_laga = self.model.get_laga()
        for i in range(2):
            if current_laga[i].abs().item() < self.laga_deadzone:
                performance_stop[i] = True

        if self.epoch % self.validate_epoch == 0 and not all(performance_stop):
            # compute performance on validation set
            with torch.no_grad():
                self.model.eval()
                dict_performance = self.model.eval_performance(self.loader_validate)
                vad_tpr = dict_performance["tpr"]
                vad_cost = dict_performance["cost"]
                performance_gap = [0, 0]
                # for laga[0]
                performance_gap[0] = (
                    0 if self.desired_cost is None
                    else abs(self.desired_cost - vad_cost)
                )
                # for laga[1]
                performance_gap[1] = (
                    0 if self.desired_tpr is None 
                    else abs(self.desired_tpr - vad_tpr)
                )
            # special waiver
            performance_gap = [
                0 if pf_stop else gap
                for (gap, pf_stop) in zip(performance_gap, performance_stop)
            ]
            if max(performance_gap) <= self.desired_tol:
                self.counter += 1
            else:
                self.counter = 0
            if self.counter >= self.patience:
                performance_stop = [True, True]

        if all(performance_stop) and lagdual_stop:
            return True

        # otherwise, keep training
        return False
